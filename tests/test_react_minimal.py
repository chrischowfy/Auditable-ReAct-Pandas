import json
import pathlib
import sys
import tempfile
import unittest

import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auditable_react_pandas.agent.react_minimal import MinimalReActAgent, StepObservation, safe_merge
from auditable_react_pandas.run import normalize_answer, select_effective_results


class MinimalReActNoModel(MinimalReActAgent):
    def __init__(self):
        self.model_name = "fake"
        self.agent_type = "MinimalReAct"
        self.load_exist = False
        self.log_dir = "/tmp/react_minimal_test"
        self.verbose = False
        self.max_steps_two = 4
        self.max_steps_three = 6
        self.max_tokens = 1200
        self.temperature = 0.0
        self.top_p = 1.0
        self.total_input_token_count = 0
        self.total_output_token_count = 0
        self.enable_relation_hints = True
        self.relation_hints_min_tables = 3
        self.enable_cell_evidence = True
        self.cell_evidence_top_k = 12
        self.cell_evidence_max_value_len = 80
        self.cell_evidence_include_rows = True
        self.enable_faithfulness_report = True
        self.faithfulness_mode = "warn_only"
        self.enable_projection_verifier = True
        self.projection_verifier_feedback = True
        self.enable_strict_recovery = True
        self.enable_one_shot_repair = False
        self.enable_safe_merge = True
        self.enable_static_checks = True
        self.enable_answer_contracts = True
        self.enable_schema_pruning = True
        self.schema_pruning_max_columns = 32
        self.schema_pruning_head_rows = 4
        self.enable_longtablebench_diagnostic_rules = False
        self.enable_candidate_selector = False
        self.candidate_selector_samples = 3
        self.candidate_selector_min_frac = 0.67
        self.candidate_selector_max_candidates = 24
        self.candidate_selector_model_name = "fake"
        self._candidate_selector_model = None
        self.execution_timeout_seconds = 30
        self._schema_pruning_diagnostics = {}
        self.run_config_hash = ""


class FakeSelectorAgent(MinimalReActNoModel):
    def __init__(self, choices):
        super().__init__()
        self.enable_candidate_selector = True
        self._choices = list(choices)

    def _query_candidate_selector(self, prompt):
        if not self._choices:
            return '{"selection": "NONE"}'
        choice = self._choices.pop(0)
        if choice == "EMAIL":
            for line in prompt.splitlines():
                if "ada@example.com" in line:
                    return json.dumps({"selection": line.split(":", 1)[0].strip()})
            return '{"selection": "NONE"}'
        return json.dumps({"selection": choice})


class MinimalReActTests(unittest.TestCase):
    def setUp(self):
        self.agent = MinimalReActNoModel()

    def test_safe_merge_aligns_join_key_dtype(self):
        left = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
        right = pd.DataFrame({"id": ["1"], "email": ["a@example.com"]})
        merged = safe_merge(left, right, on="id")
        self.assertEqual(merged["email"].tolist(), ["a@example.com"])

    def test_safe_merge_aligns_multi_key_dtype(self):
        left = pd.DataFrame({"building": ["Lambeau"], "room": [348], "course": ["A"]})
        right = pd.DataFrame({"building": ["Lambeau"], "room": ["348"], "capacity": [60]})
        merged = safe_merge(left, right, on=["building", "room"])
        self.assertEqual(merged["capacity"].tolist(), [60])

    def test_email_contract_projection(self):
        contract = self.agent.infer_contract("What is the email of candidate Jane?", table_count=2)
        structured = {
            "columns": ["candidate_details", "email_address"],
            "rows": [["Jane", "jane@example.com"]],
            "shape": [1, 2],
        }
        status, error = self.agent.check_contract("What is the email of candidate Jane?", contract, structured)
        self.assertEqual((status, error), ("PASS", ""))
        self.assertEqual(
            self.agent.deterministic_project("What is the email of candidate Jane?", contract, structured),
            "jane@example.com",
        )

    def test_contract_does_not_treat_phone_number_as_count(self):
        contract = self.agent.infer_contract(
            "What is the last name and phone number of the staff member who handled the most complaints?",
            table_count=3,
        )
        self.assertEqual(contract.answer_type, "text")
        self.assertTrue(contract.multi_field)
        self.assertIn("phone", contract.target_terms)

    def test_plural_which_states_is_list_completion_not_single_rank(self):
        contract = self.agent.infer_contract(
            "Which states have colleges that accepted at least one player based on their tryout decision?",
            table_count=3,
        )

        self.assertEqual(contract.operation, "list_completion")
        self.assertFalse(contract.requires_single)
        self.assertIn("state", contract.target_terms)

    def test_rank_cue_still_requires_single_rank_contract(self):
        contract = self.agent.infer_contract(
            "Which state has the highest number of accepted players?",
            table_count=3,
        )

        self.assertEqual(contract.operation, "rank")
        self.assertTrue(contract.requires_single)

    def test_extreme_metric_question_is_numeric_contract(self):
        contract = self.agent.infer_contract(
            "What is the maximum fastest lap speed in the Monaco Grand Prix in 2008?",
            table_count=2,
        )

        self.assertEqual(contract.answer_type, "number")
        self.assertEqual(contract.operation, "rank")
        structured = {"columns": ["fastestLapSpeed"], "rows": [["156.789"]], "shape": [1, 1]}
        self.assertEqual(
            self.agent.check_contract(
                "What is the maximum fastest lap speed in the Monaco Grand Prix in 2008?",
                contract,
                structured,
            ),
            ("PASS", ""),
        )
        self.assertEqual(
            self.agent.deterministic_project(
                "What is the maximum fastest lap speed in the Monaco Grand Prix in 2008?",
                contract,
                structured,
            ),
            "156.789",
        )

    def test_candidate_selector_candidates_include_numeric_extreme_projection(self):
        query = "What is the maximum fastest lap speed in the Monaco Grand Prix in 2008?"
        contract = self.agent.infer_contract(query, table_count=2)
        observations = [StepObservation(
            step_id=1,
            action="pandas_code",
            status="PASS",
            structured_result={
                "columns": ["driver", "fastest_lap_speed"],
                "rows": [["Felipe Massa", 153.152]],
                "projection_rows": [["Felipe Massa", 153.152]],
                "shape": [1, 2],
            },
        )]

        candidates = self.agent._build_candidate_selector_candidates(query, contract, observations, "153.152")

        self.assertTrue(any(candidate["answer"] == "153.152" for candidate in candidates))

    def test_candidate_selector_requested_id_multifield_candidate_contains_id(self):
        query = "What are the organization id and organization name for the selected grant?"
        contract = self.agent.infer_contract(query, table_count=2)
        observations = [StepObservation(
            step_id=1,
            action="pandas_code",
            status="PASS",
            structured_result={
                "columns": ["organization_id", "organization_name", "grant_total"],
                "rows": [["ORG-7", "North Lab", 4]],
                "projection_rows": [["ORG-7", "North Lab", 4]],
                "shape": [1, 3],
            },
        )]

        candidates = self.agent._build_candidate_selector_candidates(query, contract, observations, "North Lab")

        self.assertTrue(any(
            "organization_id" in candidate["selected_columns"]
            and "organization_name" in candidate["selected_columns"]
            for candidate in candidates
        ))

    def test_candidate_selector_prioritizes_requested_id_name_combo_in_wide_result(self):
        query = "Which team offers the lowest average salary? Give me the name and id of the team."
        contract = self.agent.infer_contract(query, table_count=2)
        filler_cols = [f"metric_{idx}" for idx in range(30)]
        columns = filler_cols + ["team_id", "name", "avg_salary"]
        row = list(range(30)) + ["ML4", "Milwaukee Brewers", 100.0]
        observations = [StepObservation(
            step_id=1,
            action="pandas_code",
            status="PASS",
            structured_result={
                "columns": columns,
                "rows": [row],
                "projection_rows": [row],
                "shape": [1, len(columns)],
            },
        )]

        candidates = self.agent._build_candidate_selector_candidates(query, contract, observations, "100.0")

        self.assertTrue(any(
            candidate["selected_columns"] == ["team_id", "name"]
            or candidate["selected_columns"] == ["name", "team_id"]
            for candidate in candidates
        ))

    def test_candidate_selector_prompt_uses_labeled_multicolumn_display(self):
        query = "Give me the name and id of the team."
        contract = self.agent.infer_contract(query, table_count=2)
        obs = StepObservation(
            step_id=1,
            action="pandas_code",
            status="PASS",
            structured_result={
                "columns": ["team_id", "name"],
                "rows": [["ML4", "Milwaukee Brewers"]],
                "projection_rows": [["ML4", "Milwaukee Brewers"]],
                "shape": [1, 2],
            },
        )

        candidate = self.agent._candidate_from_columns(query, contract, obs, ["team_id", "name"], "column_combo")
        prompt = self.agent._build_candidate_selector_prompt(query, [candidate])

        self.assertIn("team_id=ML4", prompt)
        self.assertIn("name=Milwaukee Brewers", prompt)

    def test_candidate_selector_full_row_question_generates_all_column_candidate(self):
        query = "Display all information for the selected candidate."
        contract = self.agent.infer_contract(query, table_count=1)
        observations = [StepObservation(
            step_id=1,
            action="pandas_code",
            status="PASS",
            structured_result={
                "columns": ["name", "email", "status"],
                "rows": [["Ada", "ada@example.com", "selected"]],
                "projection_rows": [["Ada", "ada@example.com", "selected"]],
                "shape": [1, 3],
            },
        )]

        candidates = self.agent._build_candidate_selector_candidates(query, contract, observations, "Ada")

        self.assertTrue(any(
            candidate["kind"] == "full_table_projection"
            and candidate["selected_columns"] == ["name", "email", "status"]
            for candidate in candidates
        ))

    def test_candidate_selector_two_of_three_votes_switches_answer(self):
        agent = FakeSelectorAgent(["EMAIL", "EMAIL", "NONE"])
        query = "What is the email for Ada?"
        contract = agent.infer_contract(query, table_count=1)
        observations = [StepObservation(
            step_id=1,
            action="pandas_code",
            status="PASS",
            structured_result={
                "columns": ["name", "email"],
                "rows": [["Ada", "ada@example.com"]],
                "projection_rows": [["Ada", "ada@example.com"]],
                "shape": [1, 2],
            },
        )]

        answer, reason, diagnostics = agent._apply_candidate_selector(
            query, contract, observations, "Ada", "contract_satisfied"
        )

        self.assertEqual(answer, "ada@example.com")
        self.assertTrue(diagnostics["accepted"])
        self.assertEqual(reason, "candidate_selector_from_step_1")
        self.assertEqual(observations[-1].action, "candidate_selector")

    def test_candidate_selector_low_confidence_keeps_original_answer(self):
        agent = FakeSelectorAgent(["EMAIL", "NONE", "C0"])
        query = "What is the email for Ada?"
        contract = agent.infer_contract(query, table_count=1)
        observations = [StepObservation(
            step_id=1,
            action="pandas_code",
            status="PASS",
            structured_result={
                "columns": ["name", "email"],
                "rows": [["Ada", "ada@example.com"]],
                "projection_rows": [["Ada", "ada@example.com"]],
                "shape": [1, 2],
            },
        )]

        answer, reason, diagnostics = agent._apply_candidate_selector(
            query, contract, observations, "Ada", "contract_satisfied"
        )

        self.assertEqual(answer, "Ada")
        self.assertFalse(diagnostics["accepted"])
        self.assertEqual(reason, "contract_satisfied")

    def test_candidate_selector_none_error_and_same_answer_are_rejected(self):
        for choices, candidates, expected_reason in [
            (
                ["NONE", "NONE", "NONE"],
                [
                    {"id": "C0", "answer": "Ada", "source_step_id": None, "selected_columns": [], "structured_result": {}},
                    {"id": "C1", "answer": "Bo", "source_step_id": 1, "selected_columns": ["name"], "structured_result": {}},
                ],
                "selector_chose_none",
            ),
            (
                ["C1", "C1", "C1"],
                [
                    {"id": "C0", "answer": "Ada", "source_step_id": None, "selected_columns": [], "structured_result": {}},
                    {"id": "C1", "answer": "Error: bad projection", "source_step_id": 1, "selected_columns": [], "structured_result": {}},
                ],
                "selected_answer_rejected",
            ),
            (
                ["C1", "C1", "C1"],
                [
                    {"id": "C0", "answer": "Ada", "source_step_id": None, "selected_columns": [], "structured_result": {}},
                    {"id": "C1", "answer": "ada", "source_step_id": 1, "selected_columns": ["name"], "structured_result": {}},
                ],
                "same_as_current_answer",
            ),
        ]:
            agent = FakeSelectorAgent(choices)
            query = "Who is selected?"
            contract = agent.infer_contract(query, table_count=1)
            observations = [StepObservation(
                step_id=1,
                action="pandas_code",
                status="PASS",
                structured_result={
                    "columns": ["name"],
                    "rows": [["Ada"]],
                    "projection_rows": [["Ada"]],
                    "shape": [1, 1],
                },
            )]
            if candidates is not None:
                agent._build_candidate_selector_candidates = lambda *args, _candidates=candidates: _candidates

            answer, reason, diagnostics = agent._apply_candidate_selector(
                query, contract, observations, "Ada", "contract_satisfied"
            )

            self.assertEqual(answer, "Ada")
            self.assertEqual(reason, "contract_satisfied")
            self.assertFalse(diagnostics["accepted"])
            self.assertEqual(diagnostics["reason"], expected_reason)

    def test_candidate_selector_loose_parser_prefers_explicit_candidate_over_prompt_none(self):
        raw = (
            "If no candidate answers the question, select NONE. Candidate answers: "
            "C0: wrong C1: right. The best candidate is C1."
        )
        self.assertEqual(self.agent._parse_candidate_selector_choice(raw), "C1")

    def test_candidate_selector_loose_parser_does_not_pick_prompt_echo(self):
        raw = "Candidate answers: C0: Ada C1: Bo. If no candidate answers, select NONE."
        self.assertEqual(self.agent._parse_candidate_selector_choice(raw), "NONE")

    def test_candidate_selector_loose_parser_accepts_conclusion_candidate(self):
        raw = (
            "C0 is only a surname. C1 gives driverId and surname. "
            "C2 is only an id. So C1 is the most exact answer."
        )
        self.assertEqual(self.agent._parse_candidate_selector_choice(raw), "C1")

    def test_shortest_length_question_projects_numeric_length(self):
        query = "Find the length of the shortest movie in the 'Drama' category."
        contract = self.agent.infer_contract(query, table_count=3)

        self.assertEqual(contract.answer_type, "number")
        structured = {"columns": ["title", "length"], "rows": [["KWAI HOMEWARD", 46]], "shape": [1, 2]}
        self.assertEqual(self.agent.check_contract(query, contract, structured), ("PASS", ""))
        self.assertEqual(self.agent.deterministic_project(query, contract, structured), "46")

    def test_projection_verifier_requires_name_and_email(self):
        query = "What are the name and email of the candidate assigned to case 4?"
        contract = self.agent.infer_contract(query, table_count=2)
        partial = {
            "columns": ["email"],
            "rows": [["ann@example.com"]],
            "shape": [1, 1],
            "truncated": False,
        }
        status, error = self.agent.check_contract(query, contract, partial)
        self.assertEqual(status, "FAIL")
        self.assertIn("missing_slot=name", error)

        complete = {
            "columns": ["candidate_details", "email"],
            "rows": [["Ann Lee", "ann@example.com"]],
            "shape": [1, 2],
            "truncated": False,
        }
        self.assertEqual(self.agent.check_contract(query, contract, complete), ("PASS", ""))
        self.assertEqual(self.agent.deterministic_project(query, contract, complete), "Ann Lee, ann@example.com")

    def test_projection_verifier_requires_address_and_email(self):
        query = "Return the address and email of the customer with the first name Linda."
        contract = self.agent.infer_contract(query, table_count=2)
        self.assertEqual([slot["name"] for slot in contract.required_output_slots], ["address", "email"])

        partial = {
            "columns": ["email"],
            "rows": [["linda@example.com"]],
            "shape": [1, 1],
            "truncated": False,
        }
        status, error = self.agent.check_contract(query, contract, partial)
        self.assertEqual(status, "FAIL")
        self.assertIn("missing_slot=address", error)

        complete = {
            "columns": ["address", "email"],
            "rows": [["692 Joliet Street", "linda@example.com"]],
            "shape": [1, 2],
            "truncated": False,
        }
        self.assertEqual(self.agent.check_contract(query, contract, complete), ("PASS", ""))
        self.assertEqual(self.agent.deterministic_project(query, contract, complete), "692 Joliet Street, linda@example.com")

    def test_projection_verifier_requires_title_id_and_description(self):
        query = "What are the title, id, and description of the movie with the greatest number of actors?"
        contract = self.agent.infer_contract(query, table_count=2)
        required_slot_names = [slot["name"] for slot in contract.required_output_slots if slot.get("required", True)]
        self.assertEqual(required_slot_names, ["title", "id", "detail"])

        partial = {
            "columns": ["title", "count"],
            "rows": [["ACADEMY DINOSAUR", 10]],
            "shape": [1, 2],
            "truncated": False,
        }
        status, error = self.agent.check_contract(query, contract, partial)
        self.assertEqual(status, "FAIL")
        self.assertIn("id", error)
        self.assertIn("detail", error)

        complete = {
            "columns": ["title", "film_id", "description"],
            "rows": [["ACADEMY DINOSAUR", 1, "A documentary"]],
            "shape": [1, 3],
            "truncated": False,
        }
        self.assertEqual(self.agent.check_contract(query, contract, complete), ("PASS", ""))
        self.assertEqual(self.agent.deterministic_project(query, contract, complete), "ACADEMY DINOSAUR, 1, A documentary")

    def test_projection_verifier_audit_only_does_not_fail_contract(self):
        self.agent.projection_verifier_feedback = False
        query = "What are the name and email of the candidate assigned to case 4?"
        contract = self.agent.infer_contract(query, table_count=2)
        partial = {
            "columns": ["email"],
            "rows": [["ann@example.com"]],
            "shape": [1, 1],
            "truncated": False,
        }

        self.assertEqual(self.agent.check_contract(query, contract, partial), ("PASS", ""))
        report = self.agent._verify_faithfulness(
            query=query,
            contract=contract,
            final_answer="ann@example.com",
            observations=[
                StepObservation(
                    step_id=1,
                    action="pandas_code",
                    status="PASS",
                    structured_result=partial,
                    contract_status="PASS",
                )
            ],
            cell_anchors=[],
            relation_anchors=[],
            finish_reason="contract_satisfied",
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["projection_complete"])
        self.assertIn("name", report["missing_target_slots"])

    def test_projection_verifier_requires_time_and_room(self):
        query = "What time and room is the database class held in?"
        contract = self.agent.infer_contract(query, table_count=2)
        partial = {
            "columns": ["room"],
            "rows": [["B12"]],
            "shape": [1, 1],
            "truncated": False,
        }
        status, error = self.agent.check_contract(query, contract, partial)
        self.assertEqual(status, "FAIL")
        self.assertIn("missing_slot=time", error)

        complete = {
            "columns": ["start_time", "room"],
            "rows": [["09:30", "B12"]],
            "shape": [1, 2],
            "truncated": False,
        }
        self.assertEqual(self.agent.check_contract(query, contract, complete), ("PASS", ""))
        self.assertEqual(self.agent.deterministic_project(query, contract, complete), "09:30, B12")

    def test_multi_field_projection_keeps_requested_ids(self):
        query = "What are the names and ids of artists with 3 or more albums?"
        contract = self.agent.infer_contract(query, table_count=2)
        structured = {
            "columns": ["name", "and_id", "artists_id"],
            "rows": [["Audioslave", 999, 8]],
            "shape": [1, 3],
        }

        self.assertEqual(self.agent.check_contract(query, contract, structured), ("PASS", ""))
        self.assertEqual(self.agent.deterministic_project(query, contract, structured), "Audioslave, 8")

    def test_list_completion_empty_result_renders_empty_list(self):
        query = "List all album titles by Deep Purple that contain the word 'Live'."
        contract = self.agent.infer_contract(query, table_count=2)
        structured = {"columns": ["title"], "rows": [], "shape": [0, 1], "truncated": False}

        self.assertEqual(self.agent.check_contract(query, contract, structured), ("FAIL", "empty structured result"))
        answer, reason = self.agent._strict_final_recovery(
            query,
            contract,
            [StepObservation(step_id=1, action="pandas_code", status="PASS", structured_result=structured)],
        )
        self.assertEqual(answer, "[]")
        self.assertEqual(reason, "strict_final_recovery_empty_list_from_step_1")

    def test_strict_recovery_prefers_non_empty_list_over_later_empty_result(self):
        query = "What are the album titles for albums containing both 'Reggae' and 'Rock' genre tracks?"
        contract = self.agent.infer_contract(query, table_count=3)
        non_empty = {"columns": ["Title"], "rows": [["Greatest Hits"]], "shape": [1, 1], "truncated": False}
        empty = {"columns": ["Title"], "rows": [], "shape": [0, 1], "truncated": False}

        answer, reason = self.agent._strict_final_recovery(
            query,
            contract,
            [
                StepObservation(step_id=1, action="pandas_code", status="PASS", structured_result=non_empty),
                StepObservation(step_id=2, action="pandas_code", status="PASS", structured_result=empty),
            ],
        )
        self.assertEqual(answer, "Greatest Hits")
        self.assertEqual(reason, "strict_final_recovery_from_step_1")

    def test_projection_verifier_rank_returns_department_not_count(self):
        query = "Which department has the maximum count of employees?"
        contract = self.agent.infer_contract(query, table_count=2)
        self.assertEqual(contract.answer_type, "text")
        self.assertEqual(contract.operation, "rank")

        count_only = {"columns": ["count"], "rows": [[7]], "shape": [1, 1], "truncated": False}
        status, error = self.agent.check_contract(query, contract, count_only)
        self.assertEqual(status, "FAIL")
        self.assertIn("missing_slot=department", error)

        dept = {"columns": ["department", "count"], "rows": [["Sales", 7]], "shape": [1, 2], "truncated": False}
        self.assertEqual(self.agent.check_contract(query, contract, dept), ("PASS", ""))
        self.assertEqual(self.agent.deterministic_project(query, contract, dept), "Sales")

    def test_contract_total_quantity_with_year_filter_is_number(self):
        contract = self.agent.infer_contract(
            "What is the total quantity in stock of devices running Android software platform in shops opened from year 2010 onwards?",
            table_count=3,
        )
        self.assertEqual(contract.answer_type, "number")
        self.assertEqual(contract.operation, "sum")

    def test_text_contract_rejects_only_id_column(self):
        contract = self.agent.infer_contract("Who is registered for statistics?", table_count=3)
        structured = {"columns": ["student_id"], "rows": [[111]], "shape": [1, 1]}
        status, error = self.agent.check_contract("Who is registered for statistics?", contract, structured)
        self.assertEqual(status, "FAIL")
        self.assertIn("ids", error)

    def test_text_contract_allows_requested_id_answer(self):
        contract = self.agent.infer_contract("What is the most recent car insurance policy ID?", table_count=2)
        structured = {"columns": ["Policy_ID"], "rows": [[218]], "shape": [1, 1]}
        status, error = self.agent.check_contract("What is the most recent car insurance policy ID?", contract, structured)
        self.assertEqual((status, error), ("PASS", ""))
        self.assertEqual(
            self.agent.deterministic_project("What is the most recent car insurance policy ID?", contract, structured),
            "218",
        )

    def test_projection_verifier_id_question_stays_specific(self):
        query = "What is the most recent car insurance policy ID?"
        contract = self.agent.infer_contract(query, table_count=2)
        verdict = self.agent.verify_projection(
            query,
            contract,
            {"columns": ["Customer_ID"], "rows": [[4]], "shape": [1, 1]},
        )
        self.assertEqual(verdict["status"], "FAIL")
        self.assertEqual(verdict["missing_slots"], ["policy_id"])

    def test_text_contract_rejects_non_target_id_column(self):
        contract = self.agent.infer_contract("What is the most recent car insurance policy ID?", table_count=2)
        structured = {"columns": ["Customer_ID"], "rows": [[4]], "shape": [1, 1]}
        status, error = self.agent.check_contract("What is the most recent car insurance policy ID?", contract, structured)
        self.assertEqual(status, "FAIL")
        self.assertIn("no plausible", error)

    def test_filter_id_mention_does_not_allow_id_answer(self):
        contract = self.agent.infer_contract("Which medicines inhibit enzyme with ID 2?", table_count=2)
        structured = {"columns": ["enzyme_id"], "rows": [[2]], "shape": [1, 1]}
        status, error = self.agent.check_contract("Which medicines inhibit enzyme with ID 2?", contract, structured)
        self.assertEqual(status, "FAIL")
        self.assertIn("ids", error)

    def test_text_contract_rejects_count_scalar(self):
        contract = self.agent.infer_contract("Which customers have both savings and checking accounts?", table_count=3)
        structured = {"columns": ["count"], "rows": [[2]], "shape": [1, 1]}
        status, error = self.agent.check_contract("Which customers have both savings and checking accounts?", contract, structured)
        self.assertEqual(status, "FAIL")
        self.assertIn("numeric", error)

    def test_aggregate_number_contract_rejects_id_list(self):
        contract = self.agent.infer_contract("How many engineer visits are associated with asset ID 3?", table_count=2)
        structured = {"columns": ["visit_id"], "rows": [[1], [9], [13]], "shape": [3, 1]}
        status, error = self.agent.check_contract("How many engineer visits are associated with asset ID 3?", contract, structured)
        self.assertEqual(status, "FAIL")
        self.assertIn("aggregate numeric", error)

    def test_text_contract_rejects_no_result_string(self):
        contract = self.agent.infer_contract("What is the activator?", table_count=3)
        structured = {"columns": ["value"], "rows": [["No activator found"]], "shape": [1, 1]}
        status, error = self.agent.check_contract("What is the activator?", contract, structured)
        self.assertEqual(status, "FAIL")
        self.assertIn("no-result", error)

    def test_rank_contract_projects_first_row(self):
        contract = self.agent.infer_contract("Which university has the highest enrollment?", table_count=2)
        structured = {
            "columns": ["university_name", "enrollment"],
            "rows": [["A", 100], ["B", 90]],
            "shape": [2, 2],
        }
        self.assertEqual(
            self.agent.deterministic_project("Which university has the highest enrollment?", contract, structured),
            "A",
        )

    def test_static_check_rejects_raw_merge(self):
        df_0 = pd.DataFrame({"id": [1]})
        df_1 = pd.DataFrame({"id": [1]})
        ok, error = self.agent._static_code_check("step_result = pd.merge(df_0, df_1, on='id')", {"df_0": df_0, "df_1": df_1}, {})
        self.assertFalse(ok)
        self.assertIn("safe_merge", error)

    def test_static_check_rejects_bad_safe_merge_key(self):
        df_0 = pd.DataFrame({"student_id": [1], "name": ["Ann"]})
        df_1 = pd.DataFrame({"course_id": [1], "title": ["Math"]})
        ok, error = self.agent._static_code_check(
            "step_result = safe_merge(df_0, df_1, on='id')",
            {"df_0": df_0, "df_1": df_1},
            {},
        )
        self.assertFalse(ok)
        self.assertIn("safe_merge on='id'", error)
        self.assertIn("closest", error)

    def test_static_check_tracks_selected_dataframe_columns(self):
        df_0 = pd.DataFrame({"id": [1], "name": ["Ann"], "email": ["a@example.com"]})
        code = "tmp = df_0[['id', 'name']]\nstep_result = tmp[['email']]"
        ok, error = self.agent._static_code_check(code, {"df_0": df_0}, {})
        self.assertFalse(ok)
        self.assertIn("email", error)

    def test_static_check_tracks_safe_merge_suffix_columns(self):
        df_0 = pd.DataFrame({"id": [1], "name": ["Ann"]})
        df_1 = pd.DataFrame({"id": [1], "name": ["CS"]})
        code = "merged = safe_merge(df_0, df_1, on='id')\nstep_result = merged[['name']]"
        ok, error = self.agent._static_code_check(code, {"df_0": df_0, "df_1": df_1}, {})
        self.assertFalse(ok)
        self.assertIn("name", error)

    def test_static_check_tracks_dataframe_column_assignment(self):
        df_0 = pd.DataFrame({"price": [10, 20], "name": ["a", "b"]})
        code = (
            "tmp = df_0.copy()\n"
            "tmp['diff'] = (tmp['price'] - 15).abs()\n"
            "step_result = tmp.sort_values('diff')[['name', 'diff']]"
        )
        ok, error = self.agent._static_code_check(code, {"df_0": df_0}, {})
        self.assertTrue(ok, error)

    def test_static_check_tracks_assign_columns(self):
        df_0 = pd.DataFrame({"price": [10, 20], "name": ["a", "b"]})
        code = (
            "tmp = df_0.assign(diff=(df_0['price'] - 15).abs())\n"
            "step_result = tmp.sort_values('diff')[['name', 'diff']]"
        )
        ok, error = self.agent._static_code_check(code, {"df_0": df_0}, {})
        self.assertTrue(ok, error)

    def test_execute_rejects_imports_and_file_io(self):
        df_0 = pd.DataFrame({"id": [1]})
        status, msg, _, _ = self.agent._execute_code(
            "import os\nstep_result = df_0",
            {"df_0": df_0},
            {},
        )
        self.assertEqual(status, "ERROR")
        self.assertIn("imports are not allowed", msg)

        status, msg, _, _ = self.agent._execute_code(
            "step_result = open('/etc/passwd').read()",
            {"df_0": df_0},
            {},
        )
        self.assertEqual(status, "ERROR")
        self.assertIn("unsafe", msg)

    def test_execute_rejects_pandas_file_io(self):
        df_0 = pd.DataFrame({"id": [1]})
        status, msg, _, _ = self.agent._execute_code(
            "df_0.to_csv('/tmp/blocked.csv')\nstep_result = df_0",
            {"df_0": df_0},
            {},
        )
        self.assertEqual(status, "ERROR")
        self.assertIn("to_csv", msg)

    def test_execute_rejects_numpy_file_io(self):
        df_0 = pd.DataFrame({"id": [1]})
        status, msg, _, _ = self.agent._execute_code(
            "step_result = np.fromfile('/etc/hostname', dtype=str)",
            {"df_0": df_0},
            {},
        )
        self.assertEqual(status, "ERROR")
        self.assertIn("fromfile", msg)

    def test_extract_json_action_recovers_loose_multiline_code(self):
        raw = """{
  "thought": "Filter rows.",
  "action": "pandas_code",
  "code": "tmp = df_0[df_0['name'] == 'Leo']
step_result = tmp[['email']]"
}"""
        action = self.agent._extract_json_action(raw)
        self.assertEqual(action["action"], "pandas_code")
        self.assertIn("tmp = df_0", action["code"])
        self.assertIn("step_result", action["code"])

    def test_extract_code_strips_unclosed_python_fence(self):
        code = self.agent._extract_code("```python\nstep_result = df_0")
        self.assertEqual(code, "step_result = df_0")

    def test_execute_canonicalizes_harmless_imports_and_pd_merge(self):
        df_0 = pd.DataFrame({"id": [1], "name": ["Ann"]})
        df_1 = pd.DataFrame({"id": ["1"], "email": ["ann@example.com"]})
        code = """import pandas as pd
import numpy as np
def safe_merge(left, right, left_on, right_on, how='inner'):
    return pd.merge(left, right, left_on=left_on, right_on=right_on, how=how)
step_result = pd.merge(df_0, df_1, left_on='id', right_on='id', how='inner')[['email']]"""
        status, msg, structured, _ = self.agent._execute_code(code, {"df_0": df_0, "df_1": df_1}, {})
        self.assertEqual(status, "PASS", msg)
        self.assertEqual(structured["rows"], [["ann@example.com"]])

    def test_execute_canonicalizes_safe_merge_import_and_escaped_newlines(self):
        code = "from safe_merge import safe_merge\\nans = 'ok'"
        status, msg, structured, _ = self.agent._execute_code(code, {"df_0": pd.DataFrame({"id": [1]})}, {})
        self.assertEqual(status, "PASS", msg)
        self.assertEqual(structured["rows"], [["ok"]])

    def test_relation_hints_include_overlap_join(self):
        self.agent.relation_hints_min_tables = 2
        dfs = {
            "df_0": pd.DataFrame({"student_id": [1, 2], "name": ["Ann", "Bob"]}),
            "df_1": pd.DataFrame({"student_id": ["1", "3"], "course": ["Math", "CS"]}),
        }
        hints = self.agent._build_relation_hints(dfs, ["students", "enrollments"])
        self.assertIn("df_0.student_id <-> df_1.student_id", hints)
        self.assertIn("sample_overlap=1", hints)

    def test_relation_evidence_includes_structured_anchor(self):
        self.agent.relation_hints_min_tables = 2
        dfs = {
            "df_0": pd.DataFrame({"candidate_id": [161], "email": ["leo@example.com"]}),
            "df_1": pd.DataFrame({"candidate_id": ["161"], "candidate_details": ["Leo"]}),
        }
        text, anchors = self.agent._build_relation_evidence(dfs, ["people", "candidates"])
        self.assertIn("df_0.candidate_id <-> df_1.candidate_id", text)
        self.assertEqual(anchors[0]["edge_id"], "r1")
        self.assertEqual(anchors[0]["left_col"], "candidate_id")
        self.assertEqual(anchors[0]["right_col"], "candidate_id")

    def test_relation_anchors_exist_when_prompt_hints_skipped(self):
        self.agent.relation_hints_min_tables = 3
        dfs = {
            "df_0": pd.DataFrame({"person_id": [161], "email": ["leo@example.com"]}),
            "df_1": pd.DataFrame({"candidate_id": ["161"], "candidate_details": ["Leo"]}),
        }
        text, anchors = self.agent._build_relation_evidence(dfs, ["people", "candidates"])
        self.assertEqual(text, "Relation hints skipped for 2 tables.")
        self.assertTrue(any(anchor["left_col"] == "person_id" and anchor["right_col"] == "candidate_id" for anchor in anchors))

    def test_relation_evidence_includes_value_overlap_foreign_key(self):
        self.agent.relation_hints_min_tables = 3
        dfs = {
            "df_0": pd.DataFrame({"EmployeeID": [1, 2, 3], "Name": ["A", "B", "C"]}),
            "df_1": pd.DataFrame({"DepartmentID": [10, 20], "Name": ["Surgery", "Medicine"]}),
            "df_2": pd.DataFrame({"Physician": ["1", "2"], "Department": ["10", "20"]}),
        }
        text, anchors = self.agent._build_relation_evidence(dfs, ["Physician", "Department", "Affiliated_With"])
        self.assertIn("df_2.Physician", text)
        self.assertTrue(
            any(
                {anchor["left_col"], anchor["right_col"]} == {"EmployeeID", "Physician"}
                and anchor["reason"] == "value-overlap"
                for anchor in anchors
            )
        )

    def test_cell_evidence_scans_beyond_schema_head(self):
        dfs = {
            "df_0": pd.DataFrame({
                "candidate_id": [1, 2, 3, 4, 5, 6],
                "candidate_details": ["Ada", "Bo", "Cy", "Dee", "Eli", "Leo"],
                "email": ["a@x.com", "b@x.com", "c@x.com", "d@x.com", "e@x.com", "leo@example.com"],
            })
        }
        evidence = self.agent._build_cell_evidence("What is the email of 'Leo'?", dfs, ["candidates"])
        self.assertIn("df_0.candidate_details row=5", evidence)
        self.assertIn('"Leo"', evidence)
        self.assertIn("leo@example.com", evidence)

    def test_cell_evidence_includes_structured_anchor(self):
        dfs = {
            "df_0": pd.DataFrame({
                "candidate_id": [1, 2, 3, 4, 5, 6],
                "candidate_details": ["Ada", "Bo", "Cy", "Dee", "Eli", "Leo"],
                "email": ["a@x.com", "b@x.com", "c@x.com", "d@x.com", "e@x.com", "leo@example.com"],
            })
        }
        anchors = self.agent._build_cell_evidence_anchors("What is the email of 'Leo'?", dfs, ["candidates"])
        self.assertEqual(anchors[0]["anchor_id"], "c1")
        self.assertEqual(anchors[0]["column"], "candidate_details")
        self.assertEqual(anchors[0]["row"], 5)
        self.assertEqual(anchors[0]["value"], "Leo")
        self.assertEqual(anchors[0]["row_context"]["email"], "leo@example.com")

    def test_cell_evidence_can_be_disabled(self):
        self.agent.enable_cell_evidence = False
        evidence = self.agent._build_cell_evidence(
            "What is the email of Leo?",
            {"df_0": pd.DataFrame({"name": ["Leo"]})},
            ["candidates"],
        )
        self.assertEqual(evidence, "Cell evidence disabled.")

    def test_prompt_includes_cell_evidence_block(self):
        contract = self.agent.infer_contract("What is the email of Leo?", table_count=1)
        prompt = self.agent._build_prompt(
            "What is the email of Leo?",
            "- candidates (df_0): ['name', 'email']",
            "Relation hints skipped for 1 tables.",
            "- df_0.name row=5 value=\"Leo\"",
            contract,
            [],
        )
        self.assertIn("Relevant cell evidence from full tables:", prompt)
        self.assertIn("df_0.name row=5", prompt)
        self.assertIn("verify the answer by pandas operations over the full DataFrames", prompt)
        self.assertIn("use .isin(...)", prompt)

    def test_prompt_allows_all_available_dataframes(self):
        contract = self.agent.infer_contract("Which student is enrolled in course C?", table_count=5)
        prompt = self.agent._build_prompt(
            "Which student is enrolled in course C?",
            "\n".join(f"- table_{i} (df_{i}): ['id']" for i in range(5)),
            "No relation hints.",
            "No relevant cell evidence found.",
            contract,
            [],
            df_names=[f"df_{i}" for i in range(5)],
        )

        self.assertIn("df_0, df_1, df_2, df_3, df_4", prompt)
        self.assertNotIn("Use only df_0, df_1, df_2, previous variables", prompt)

    def test_schema_pruning_caps_wide_table_prompt_without_pruning_dfs(self):
        query = "What is the name of the customer who placed the earliest order?"
        noise_cols = [f"noise_{i}" for i in range(1497)]
        columns = noise_cols[:700] + ["Customer_ID", "Customer_Name", "Order_Date"] + noise_cols[700:]
        rows = [
            [f"x{i}" for i in range(700)] + [1, "Ada", "2020-01-02"] + [f"y{i}" for i in range(797)],
            [f"u{i}" for i in range(700)] + [2, "Leo", "2019-01-02"] + [f"v{i}" for i in range(797)],
        ]
        data = {
            "table_names": ["orders"],
            "tables": [{"table_columns": columns, "table_content": rows}],
        }
        contract = self.agent.infer_contract(query, table_count=1)
        dfs, schema_text = self.agent._build_dfs(data, query=query, contract=contract)

        self.assertEqual(len(dfs["df_0"].columns), 1500)
        diag = self.agent._schema_pruning_diagnostics["tables"][0]
        self.assertTrue(diag["pruned"])
        self.assertEqual(diag["selected_column_count"], 32)
        self.assertEqual(diag["hidden_column_count"], 1468)
        self.assertIn("Customer_ID", diag["selected_columns"])
        self.assertIn("Customer_Name", diag["selected_columns"])
        self.assertIn("Order_Date", diag["selected_columns"])
        self.assertIn("schema-pruned: 1468 hidden columns", schema_text)
        self.assertLess(len(schema_text), 12000)

    def test_schema_pruning_reveals_query_relevant_hidden_columns(self):
        self.agent.schema_pruning_max_columns = 1
        query = "What are the customer name and customer email for order 4?"
        data = {
            "table_names": ["orders"],
            "tables": [{
                "table_columns": ["order_id", "customer_name", "customer_email", "noise"],
                "table_content": [[4, "Ada", "ada@example.com", "x"]],
            }],
        }
        contract = self.agent.infer_contract(query, table_count=1)
        _, schema_text = self.agent._build_dfs(data, query=query, contract=contract)
        diag = self.agent._schema_pruning_diagnostics["tables"][0]

        self.assertEqual(diag["selected_column_count"], 1)
        self.assertTrue(diag["revealed_hidden_columns"])
        self.assertIn("Query-relevant hidden columns:", schema_text)

    def test_schema_pruning_does_not_read_column_stress_gold_columns(self):
        self.agent.schema_pruning_max_columns = 5
        query = "Which customer placed the earliest order?"
        columns = [f"noise_{i}" for i in range(20)] + ["private_gold_slot", "Customer_ID", "Order_Date"]
        rows = [[f"x{i}" for i in range(20)] + ["Ada", 1, "2020-01-01"]]
        data = {
            "table_names": ["orders"],
            "column_stress": {"tables_meta": [{"gold_columns": ["private_gold_slot"]}]},
            "tables": [{"table_columns": columns, "table_content": rows}],
        }
        contract = self.agent.infer_contract(query, table_count=1)
        self.agent._build_dfs(data, query=query, contract=contract)

        selected = self.agent._schema_pruning_diagnostics["tables"][0]["selected_columns"]
        self.assertIn("Customer_ID", selected)
        self.assertIn("Order_Date", selected)
        self.assertNotIn("private_gold_slot", selected)

    def test_relation_sampling_includes_tail_values(self):
        left = pd.Series([f"left_{i}" for i in range(245)] + ["join_key", "tail_key"])
        right = pd.Series([f"right_{i}" for i in range(245)] + ["join_key", "other_tail"])

        self.assertIn("join_key", self.agent._sample_values(left, limit=40))
        self.assertIn("join_key", self.agent._sample_values(right, limit=40))

    def test_failure_schema_reveal_adds_relevant_hidden_columns(self):
        self.agent.schema_pruning_max_columns = 1
        query = "What is the customer email for order 4?"
        data = {
            "table_names": ["orders"],
            "tables": [{
                "table_columns": ["order_id", "customer_email", "noise"],
                "table_content": [[4, "ada@example.com", "x"]],
            }],
        }
        contract = self.agent.infer_contract(query, table_count=1)
        dfs, schema_text = self.agent._build_dfs(data, query=query, contract=contract)
        observations = [
            StepObservation(
                step_id=1,
                action="pandas_code",
                status="ERROR",
                error="KeyError: customer_email not in columns",
                contract_status="FAIL",
            )
        ]

        reveal = self.agent._build_failure_schema_reveal(query, dfs, ["orders"], contract, observations)
        expanded = self.agent._append_schema_reveal(schema_text, reveal)

        self.assertIn("Additional hidden columns revealed after failed step:", reveal)
        self.assertIn("customer_email", expanded)

    def test_structured_projection_keeps_more_than_preview_rows(self):
        query = "Which states have accepted players?"
        contract = self.agent.infer_contract(query, table_count=2)
        structured = self.agent._object_to_structured(pd.Series([f"S{i}" for i in range(25)], name="state"))

        self.assertEqual(len(structured["rows"]), 20)
        self.assertEqual(len(structured["projection_rows"]), 25)
        self.assertEqual(self.agent.check_contract(query, contract, structured), ("PASS", ""))
        answer = self.agent.deterministic_project(query, contract, structured)
        self.assertIn("S24", answer)
        self.assertIn("projection_row_count", StepObservation(1, "pandas_code", structured_result=structured).to_log()["structured_result"])
        self.assertNotIn("projection_rows", StepObservation(1, "pandas_code", structured_result=structured).to_log()["structured_result"])

    def test_schema_pruning_keeps_small_tables_unpruned(self):
        data = {
            "table_names": ["customers"],
            "tables": [{
                "table_columns": ["customer_id", "name", "city"],
                "table_content": [[1, "Ada", "Paris"]],
            }],
        }
        contract = self.agent.infer_contract("What is the city of Ada?", table_count=1)
        _, schema_text = self.agent._build_dfs(data, query="What is the city of Ada?", contract=contract)
        diag = self.agent._schema_pruning_diagnostics["tables"][0]

        self.assertFalse(diag["pruned"])
        self.assertEqual(diag["selected_columns"], ["customer_id", "name", "city"])
        self.assertNotIn("schema-pruned", schema_text)

    def test_projection_avoids_empty_columns(self):
        contract = self.agent.infer_contract("Which movies have ratings without a rating date?", table_count=2)
        structured = {
            "columns": ["title", "ratingDate"],
            "rows": [["Snow White", None], ["Raiders of the Lost Ark", None]],
            "shape": [2, 2],
            "truncated": False,
        }
        status, error = self.agent.check_contract("Which movies have ratings without a rating date?", contract, structured)
        self.assertEqual((status, error), ("PASS", ""))
        self.assertEqual(
            self.agent.deterministic_project("Which movies have ratings without a rating date?", contract, structured),
            "Snow White, Raiders of the Lost Ark",
        )

    def test_projection_verifier_preserves_table_like_multi_column_answer(self):
        query = "List the name and email for all selected candidates."
        contract = self.agent.infer_contract(query, table_count=2)
        structured = {
            "kind": "dataframe",
            "columns": ["name", "email", "candidate_id"],
            "rows": [["Ann", "ann@example.com", 1], ["Bob", "bob@example.com", 2]],
            "shape": [2, 3],
            "truncated": False,
        }
        self.assertEqual(self.agent.check_contract(query, contract, structured), ("PASS", ""))
        self.assertEqual(
            self.agent.deterministic_project(query, contract, structured),
            "Ann, ann@example.com; Bob, bob@example.com",
        )

    def test_faithfulness_report_passes_supported_execution(self):
        self.agent.relation_hints_min_tables = 2
        query = "What is the email address of the person whose corresponding candidate detail is 'Leo'?"
        dfs = {
            "df_0": pd.DataFrame({"candidate_id": [161], "email": ["leo@example.com"]}),
            "df_1": pd.DataFrame({"candidate_id": ["161"], "candidate_details": ["Leo"]}),
        }
        _, relation_anchors = self.agent._build_relation_evidence(dfs, ["people", "candidates"])
        cell_anchors = self.agent._build_cell_evidence_anchors(query, dfs, ["people", "candidates"])
        contract = self.agent.infer_contract(query, table_count=2)
        structured = {
            "kind": "dataframe",
            "columns": ["candidate_details", "email"],
            "rows": [["Leo", "leo@example.com"]],
            "shape": [1, 2],
            "truncated": False,
        }
        report = self.agent._verify_faithfulness(
            query=query,
            contract=contract,
            final_answer="leo@example.com",
            observations=[
                StepObservation(
                    step_id=1,
                    action="pandas_code",
                    code="step_result = safe_merge(df_0, df_1, on='candidate_id')[['candidate_details', 'email']]",
                    status="PASS",
                    structured_result=structured,
                    contract_status="PASS",
                )
            ],
            cell_anchors=cell_anchors,
            relation_anchors=relation_anchors,
            finish_reason="contract_satisfied",
        )
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["answer_from_execution"])
        self.assertTrue(report["entity_anchor_supported"])
        self.assertTrue(report["join_path_supported"])
        self.assertIn("c1", report["used_cell_anchors"])
        self.assertEqual(report["used_relation_anchors"], ["r1"])

    def test_faithfulness_report_warns_unsupported_join(self):
        query = "What is the email of 'Leo'?"
        contract = self.agent.infer_contract(query, table_count=2)
        structured = {
            "kind": "dataframe",
            "columns": ["email"],
            "rows": [["leo@example.com"]],
            "shape": [1, 1],
            "truncated": False,
        }
        report = self.agent._verify_faithfulness(
            query=query,
            contract=contract,
            final_answer="leo@example.com",
            observations=[
                StepObservation(
                    step_id=1,
                    action="pandas_code",
                    code="step_result = safe_merge(df_0, df_1, left_on='bad_id', right_on='other_id')[['email']]",
                    status="PASS",
                    structured_result=structured,
                    contract_status="PASS",
                )
            ],
            cell_anchors=[{"anchor_id": "c1", "value": "Leo", "matched": ["quoted_exact:Leo"]}],
            relation_anchors=[],
            finish_reason="contract_satisfied",
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["join_path_supported"])
        self.assertIn("safe_merge_join_key_not_supported_by_relation_anchor", report["reasons"])

    def test_faithfulness_report_allows_same_key_derived_merge(self):
        query = "Which departments are in A&SCI?"
        contract = self.agent.infer_contract(query, table_count=3)
        structured = {
            "kind": "dataframe",
            "columns": ["DEPT_NAME"],
            "rows": [["English"], ["History"]],
            "shape": [2, 1],
            "truncated": False,
        }
        report = self.agent._verify_faithfulness(
            query=query,
            contract=contract,
            final_answer="English, History",
            observations=[
                StepObservation(
                    step_id=1,
                    action="pandas_code",
                    code="step_result = safe_merge(step_result, df_0, left_on='DEPT_NAME', right_on='DEPT_NAME')[['DEPT_NAME']]",
                    status="PASS",
                    structured_result=structured,
                    contract_status="PASS",
                )
            ],
            cell_anchors=[],
            relation_anchors=[],
            finish_reason="contract_satisfied",
        )
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["join_path_supported"])

    def test_strict_final_recovery_requires_contract_pass(self):
        contract = self.agent.infer_contract("Who is registered for statistics?", table_count=2)
        bad = {
            "columns": ["student_id"],
            "rows": [[111]],
            "shape": [1, 1],
        }
        answer, reason = self.agent._strict_final_recovery(
            "Who is registered for statistics?",
            contract,
            [
                # This would have been accepted by the old loose best-available projection.
                StepObservation(
                    step_id=1,
                    action="pandas_code",
                    status="PASS",
                    structured_result=bad,
                )
            ],
        )
        self.assertEqual((answer, reason), ("", ""))

    def test_truncated_structured_result_requires_narrowing(self):
        contract = self.agent.infer_contract("Which names are registered?", table_count=3)
        structured = {
            "columns": ["name"],
            "rows": [[f"name_{i}"] for i in range(20)],
            "shape": [21, 1],
            "truncated": True,
        }
        status, error = self.agent.check_contract("Which names are registered?", contract, structured)
        self.assertEqual(status, "FAIL")
        self.assertIn("truncated", error)

    def test_load_existing_requires_matching_config_hash(self):
        self.agent.load_exist = True
        self.agent.run_config_hash = "new"
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "item.json"
            path.write_text(json.dumps({"answer": "old", "run_config_hash": "old"}))
            self.assertIsNone(self.agent._load_existing_result(str(path)))
            path.write_text(json.dumps({"answer": "new", "run_config_hash": "new"}))
            self.assertEqual(self.agent._load_existing_result(str(path))["answer"], "new")

    def test_select_effective_results_uses_same_vote_unit_for_sc(self):
        rows = [
            {"id": "q1", "answer": "A", "label": "A", "sc_id": 0},
            {"id": "q1", "answer": "B", "label": "A", "sc_id": 1},
            {"id": "q1", "answer": "B", "label": "A", "sc_id": 2},
            {"id": "q2", "answer": "C", "label": "C", "sc_id": 0},
        ]
        effective = select_effective_results(rows)
        self.assertEqual(len(effective), 2)
        self.assertEqual(effective[0]["answer"], "B")
        self.assertEqual(effective[0]["sc_raw_run_count"], 3)
        self.assertEqual(effective[0]["sc_vote_count"], 2)

    def test_normalize_answer_ignores_list_order_and_number_format(self):
        self.assertEqual(normalize_answer("AZ, LA"), normalize_answer("LA, AZ"))
        self.assertEqual(normalize_answer("115897.0"), normalize_answer("115897"))

    def test_extract_code_strips_plain_language_tag(self):
        self.assertEqual(
            self.agent._extract_code("python\nstep_result = df_0[['email']]"),
            "step_result = df_0[['email']]",
        )

    def test_execute_code_returns_structured_result(self):
        df_0 = pd.DataFrame({"id": [1], "email": ["a@example.com"]})
        status, msg, structured, memory = self.agent._execute_code(
            "step_result = df_0[['email']]",
            {"df_0": df_0},
            {},
        )
        self.assertEqual(status, "PASS")
        self.assertEqual(structured["columns"], ["email"])
        self.assertEqual(structured["rows"], [["a@example.com"]])
        self.assertIn("step_result", memory)


if __name__ == "__main__":
    unittest.main()

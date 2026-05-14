from app.src.core.tools.phases.control import stop_autonomous_run, transition_phase
from app.src.core.tools.phases.hypothesize_tools.tools import (
    hypothesize_propose_mechanism,
    hypothesize_rank_mechanism,
)
from app.src.core.tools.phases.research_tools.tools import (
    fetch_paper_from_link,
    get_pubmed_abstract,
    get_trial_details,
    research_scan_literature,
    search_clinical_trials,
    search_pubmed,
)
from app.src.core.tools.phases.synthesize_tools.tools import (
    synthesize_define_next_steps,
    synthesize_generate_candidate_summary,
)
from app.src.core.tools.phases.test_tools.tools import (
    test_run_in_silico_trial,
    test_run_safety_screen,
)
from app.src.core.tools.hypothesis_bank.tools import (
    add_to_hypothesis_bank,
    search_hypothesis_bank,
    get_hypothesis_by_id,
    list_hypotheses,
    update_hypothesis_scores,
)
from app.src.core.tools.base_tools import all_base_tools

hypothesis_tools = [
    add_to_hypothesis_bank,
    search_hypothesis_bank,
    get_hypothesis_by_id,
    list_hypotheses,
    update_hypothesis_scores,
]

PHASE_TOOLS = {
    "research": [
        research_scan_literature,
        fetch_paper_from_link,
        search_clinical_trials,
        get_trial_details,
        search_pubmed,
        get_pubmed_abstract,
        transition_phase,
        *hypothesis_tools,
        *all_base_tools,
    ],
    "hypothesize": [
        hypothesize_propose_mechanism,
        hypothesize_rank_mechanism,
        transition_phase,
        *hypothesis_tools,
        *all_base_tools,
    ],
    "test": [
        test_run_in_silico_trial,
        test_run_safety_screen,
        transition_phase,
        *hypothesis_tools,
        *all_base_tools,
    ],
    "synthesize": [
        synthesize_generate_candidate_summary,
        synthesize_define_next_steps,
        stop_autonomous_run,
        transition_phase,
        *hypothesis_tools,
        *all_base_tools,
    ],
}

def get_all_phase_tools() -> list:
    tool_names = set()
    flattened_tools = []
    for phase_tools in PHASE_TOOLS.values():
        for tool_obj in phase_tools:
            if tool_obj.name in tool_names:
                continue
            tool_names.add(tool_obj.name)
            flattened_tools.append(tool_obj)
    return flattened_tools
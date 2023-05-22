import json
import sys

import pandas as pd
import streamlit as st


def parse_concatenated_json(json_str: str):
    # https://stackoverflow.com/questions/36967236/parse-multiple-json-objects-that-are-in-one-line
    decoder = json.JSONDecoder()
    pos = 0
    objs = []
    while pos < len(json_str):
        json_str = json_str[pos:].strip()
        if not json_str:
            break  # Blank line case
        obj, pos = decoder.raw_decode(json_str)
        objs.append(obj)

    return objs


@st.cache
def load_data(file: str):
    return parse_concatenated_json(open(file, "r").read())


if len(sys.argv) == 1:
    raise Exception("usage: `streamlit run scripts/view_telemetry.py <JSON log file>`")

file = sys.argv[1]

episodes = load_data(file)

is_fsm = any("fsm_current_stage" in step for step in episodes[0]["steps"])

st.sidebar.subheader("Select Episode:")
episode_num = st.sidebar.selectbox("Episode", list(range(1, len(episodes) + 1)))

st.sidebar.subheader("Select Filters:")

has_messages = episodes[0]["steps"][0]["messages"] is not None

show_observations = st.sidebar.checkbox("Show Observations", True)
show_messages = st.sidebar.checkbox("Show Messages", True) if has_messages else False
show_actions = st.sidebar.checkbox("Show Actions", True)
show_rewards = st.sidebar.checkbox("Show Rewards", True)
show_terminations = st.sidebar.checkbox("Show Terminations", False)
show_truncations = st.sidebar.checkbox("Show Truncations", False)
show_infos = st.sidebar.checkbox("Show Infos", False)
show_metrics = st.sidebar.checkbox("Show Metrics", False)

show_fsm = is_fsm and st.sidebar.checkbox("Show FSM Transitions", True)

st.sidebar.subheader("Options:")

pretty_print = st.sidebar.checkbox("Pretty Print", True)

st.title("Phantom Telemetry Viewer")

episode = episodes[episode_num - 1]

for i, step in enumerate(episode["steps"]):
    with st.expander(f"Step {i}/{len(episode['steps'])-1}", expanded=i < 2):
        if show_observations and "observations" in step:
            st.subheader("Observations:")
            if len(step["observations"]) > 0:
                for agent, obs in step["observations"].items():
                    st.text(f"{agent}:")
                    st.code(json.dumps(obs, indent=4 if pretty_print else None))
            else:
                st.text("None")

        if show_messages and len(step["messages"]) > 0:
            st.subheader("Messages:")
            if len(step["messages"]) > 0:
                df = pd.DataFrame(step["messages"])
                df.columns = ["Sender ID", "Receiver ID", "Payload"]
                st.table(df)
            else:
                st.text("None")

        if show_actions and "actions" in step:
            st.subheader("Actions:")
            if len(step["actions"]) > 0:
                for agent, act in step["actions"].items():
                    st.text(f"{agent}:")
                    st.code(json.dumps(act, indent=4 if pretty_print else None))
            else:
                st.text("None")

        if show_rewards and "rewards" in step:
            st.subheader("Rewards:")
            if len(step["rewards"]) > 0:
                df = pd.DataFrame(step["rewards"].items())
                df.columns = ["Agent", "Reward"]
                st.table(df)
            else:
                st.text("None")

        if show_terminations and "terminations" in step:
            st.subheader("Terminations:")

            if len(step["terminations"]) > 0:
                st.code(step["terminations"])
            else:
                st.text("None")

        if show_truncations and "truncations" in step:
            st.subheader("Truncations:")

            if len(step["truncations"]) > 0:
                st.code(step["truncations"])
            else:
                st.text("None")

        if show_infos and "infos" in step:
            st.subheader("Infos:")
            infos = 0
            for agent, info in step["infos"].items():
                if info is not None and info != {}:
                    st.text(f"{agent}:")
                    st.code(json.dumps(info, indent=4 if pretty_print else None))
                    infos += 1

            if infos == 0:
                st.text("None")

        if show_metrics and len(step["metrics"]) > 0:
            st.subheader("Metrics:")
            df = pd.DataFrame(step["metrics"].items())
            df.columns = ["Metric", "Value"]
            st.table(df)

        if show_fsm and "fsm_current_stage" in step and "fsm_next_stage" in step:
            st.subheader("FSM Transition:")
            st.code(f"{step['fsm_current_stage']} --> {step['fsm_next_stage']}")

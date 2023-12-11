import json
import sys

import pandas as pd
import streamlit as st

from utils import parse_concatenated_json


@st.cache_data
def load_data(file: str):
    return parse_concatenated_json(open(file, "r").read())


if len(sys.argv) == 1:
    # TODO:
    raise Exception("usage: `streamlit run scripts/view_telemetry.py <JSON log file>`")

st.set_page_config(page_title="Phantom Telemetry Viewer")

file = sys.argv[1]
episodes = load_data(file)

is_fsm = any("fsm_current_stage" in step for step in episodes[0]["steps"])

st.sidebar.subheader("Select Episode/Step:")

# episode_num = st.sidebar.selectbox("Episode", list(range(1, len(episodes) + 1)))
if len(episodes) == 1:
    episode_num = 1
    st.sidebar.text("Episode = 1/1")
else:
    episode_num = st.sidebar.slider("Episode", 1, len(episodes), 1)

episode = episodes[episode_num - 1]

# step_num = st.sidebar.selectbox("Step", list(range(1, len(episodes) + 1)))
if len(episode["steps"]) == 1:
    step_num = 0
    st.sidebar.text("Step = 1/1")
else:
    step_num = st.sidebar.slider("Step", 0, len(episode["steps"]) - 1, 1)

step = episode["steps"][step_num]

st.sidebar.subheader("Options:")

filter_string = st.sidebar.text_input("Filter", "")

pretty_print = st.sidebar.checkbox("Pretty Print", False)

st.title("Phantom Telemetry Viewer")

tab_names = [
    "Observations",
    "Messages",
    "Actions",
    "Rewards",
    "Terminations",
    "Truncations",
    "Infos",
    "Metrics",
]

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(tab_names)

with tab1:
    if "observations" in step and len(step["observations"]) > 0:
        for agent, obs in step["observations"].items():
            if filter_string == "" or filter_string in agent:
                st.text(f"{agent}:")
                st.code(json.dumps(obs, indent=4 if pretty_print else None))
    else:
        st.text("None")

with tab2:
    senders = set(msg["sender_id"] for msg in step["messages"])
    receivers = set(msg["receiver_id"] for msg in step["messages"])
    msg_types = set(msg["type"] for msg in step["messages"])

    col1, col2, col3 = st.columns(3)

    with col1:
        sender = st.selectbox("Sender", ["All"] + sorted(list(senders)))

    with col2:
        receiver = st.selectbox("Receiver", ["All"] + sorted(list(receivers)))

    with col3:
        msg_type = st.selectbox("Type", ["All"] + sorted(list(msg_types)))

    msgs = [
        msg
        for msg in step["messages"]
        if (sender == "All" or msg["sender_id"] == sender)
        and (receiver == "All" or msg["receiver_id"] == receiver)
        and (msg_type == "All" or msg["type"] == msg_type)
    ]

    st.text(f"Showing {len(msgs)} of {len(step['messages'])} messages")
    st.divider()

    for msg in msgs:
        st.text(f"{msg['sender_id']} --> {msg['receiver_id']} ({msg['type']}):")
        st.code(json.dumps(msg["payload"], indent=4 if pretty_print else None))


with tab3:
    if "actions" in step and len(step["actions"]) > 0:
        for agent, act in step["actions"].items():
            if filter_string == "" or filter_string in agent:
                st.text(f"{agent}:")
                st.code(json.dumps(act, indent=4 if pretty_print else None))
    else:
        st.text("None")

with tab4:
    if "rewards" in step and len(step["rewards"]) > 0:
        df = pd.DataFrame(step["rewards"].items(), columns=["Agent", "Reward"])
        df = df[df["Agent"].str.contains(filter_string)]

        if len(df) == 0:
            st.text("None")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.text("None")

with tab5:
    if "terminations" in step and len(step["terminations"]) > 0:
        st.code(step["terminations"])
    else:
        st.text("None")

with tab6:
    if "truncations" in step and len(step["truncations"]) > 0:
        st.code(step["truncations"])
    else:
        st.text("None")

with tab7:
    infos = 0
    if "infos" in step:
        for agent, info in step["infos"].items():
            if (
                info is not None
                and info != {}
                and (filter_string == "" or filter_string in agent)
            ):
                st.text(f"{agent}:")
                st.code(json.dumps(info, indent=4 if pretty_print else None))
                infos += 1

    if infos == 0:
        st.text("None")

with tab8:
    if step["metrics"] != {}:
        df = pd.DataFrame(step["metrics"].items(), columns=["Metric", "Value"])
        df = df[df["Metric"].str.contains(filter_string)]

        if len(df) == 0:
            st.text("None")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.text("None")

if "fsm_current_stage" in step and "fsm_next_stage" in step:
    st.subheader("FSM Transition:")
    st.code(f"{step['fsm_current_stage']} --> {step['fsm_next_stage']}")

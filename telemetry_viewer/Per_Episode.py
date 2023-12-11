import json
import sys

import networkx as nx
import matplotlib.pyplot as plt
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

st.sidebar.subheader("Select Episode:")

# episode_num = st.sidebar.selectbox("Episode", list(range(1, len(episodes) + 1)))
if len(episodes) == 1:
    episode_num = 1
    st.sidebar.text("Episode = 1/1")
else:
    episode_num = st.sidebar.slider("Episode", 1, len(episodes), 1)

episode = episodes[episode_num - 1]

st.sidebar.subheader("Options:")

pretty_print = st.sidebar.checkbox("Pretty Print", False)

st.title("Phantom Telemetry Viewer")

tab_names = ["Environment", "Agents", "Network"]

tab1, tab2, tab3 = st.tabs(tab_names)

with tab1:
    st.text("Class:")
    st.code(episode["environment"]["class"])

    st.text("Type:")
    if episode["environment"]["type"] is not None:
        st.code(
            json.dumps(
                episode["environment"]["type"], indent=4 if pretty_print else None
            )
        )
    else:
        st.code("No Type/Supertype")

    st.text("Num Steps:")
    st.code(episode["environment"]["num_steps"])

with tab2:
    agent_classes = set(agent["class"] for agent in episode["agents"])

    for agent_class in agent_classes:
        agents = [agent for agent in episode["agents"] if agent["class"] == agent_class]

        label = f"{agent_class} ({len(agents)})"

        if agents[0]["strategic"]:
            label += " [strategic]"

        with st.expander(label, expanded=False):
            for agent in agents:
                st.text(agent["id"])
                if agent["type"] is not None:
                    st.code(
                        json.dumps(agent["type"], indent=4 if pretty_print else None)
                    )
                else:
                    st.code("No Type/Supertype")

    if len(episode["agents"]) == 0:
        st.text("None")

with tab3:
    DISPLAY_ARGS = dict(
        with_labels=True,
        arrows=False,
        edge_color="grey",
        node_color="#add8e6",
        font_size=8,
    )

    G = nx.DiGraph()

    for edge in episode["connections"]:
        G.add_edge(*edge)

    subgraph = st.selectbox("Subgraph:", ["<All>"] + list(G.nodes))

    if subgraph != "<All>":
        G = G.subgraph(list(nx.neighbors(G, subgraph)) + [subgraph])

    fig, ax = plt.subplots()
    nx.draw(G, nx.shell_layout(G), **DISPLAY_ARGS)
    st.pyplot(fig)

import streamlit as st
from ..utils import get_visualization_figure


def sample_dataset(dataset, classes):
    mesh, label = next(iter(dataset))
    figure = get_visualization_figure(mesh[0], classes[label.numpy()[0]])
    st.plotly_chart(figure, use_container_width=True)

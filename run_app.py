import pickle
from glob import glob
import streamlit as st
from pointnet.app import sample_dataset
from pointnet.dataset import get_dataset, download_dataset


def run_app(classes, train_tfrecords, test_tfrecords):
    train_dataset = get_dataset(train_tfrecords, buffer_size=2048, batch_size=1, augment=False)
    test_dataset = get_dataset(test_tfrecords, buffer_size=2048, batch_size=1, augment=False)

    st.markdown('<h1>PointNet Classifier</h1></hr>', unsafe_allow_html=True)

    st.markdown('<h2>Training Samples</h2>', unsafe_allow_html=True)
    sample_dataset(train_dataset, classes)
    st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown('<h2>Test Samples</h2>', unsafe_allow_html=True)
    sample_dataset(test_dataset, classes)
    st.markdown('<hr>', unsafe_allow_html=True)


if __name__ == '__main__':
    with open('./tfrecords/classes.pkl', 'rb') as infile:
        classes = pickle.load(infile)
    run_app(classes, glob('./tfrecords/train*'), glob('./tfrecords/test*'))

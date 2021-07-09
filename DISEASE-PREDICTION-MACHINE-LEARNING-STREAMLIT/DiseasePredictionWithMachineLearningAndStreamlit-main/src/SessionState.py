from streamlit.report_thread import get_report_ctx
import streamlit as st


class SessionState(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


@st.cache(allow_output_mutation=True)
def get_session(id, **kwargs):
    return SessionState(**kwargs)


def get(**kwargs):
    ctx = get_report_ctx()
    id = ctx.session_id
    return get_session(id, **kwargs)

class DialogState:
    START = "start"
    WAIT_CITY = "wait_city"


user_states = {}


def get_state(user_id):
    return user_states.get(user_id, DialogState.START)


def set_state(user_id, state):
    user_states[user_id] = state
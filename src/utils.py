import datetime

def time_s() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

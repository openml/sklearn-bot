from time import gmtime, strftime


def get_time():
    """
    Returns a string representing the time, to be used in string output to the
    stdout and stderr

    Returns
    -------
    time: str
        A string representing the time
    """
    return strftime("[%Y-%m-%d %H:%M:%S]", gmtime())

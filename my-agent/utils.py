def calculate_average(numbers):
    """Return the mean of `numbers`, or None if there are none."""
    if not numbers:
        return None
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)


def get_user_name(user):
    """Return the user's name in uppercase, or None if it's missing."""
    if not user:
        return None
    name = user.get("name")
    if not isinstance(name, str):
        return None
    return name.upper()

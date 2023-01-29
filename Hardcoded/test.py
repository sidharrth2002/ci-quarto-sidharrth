# there are 4 elements in a list
# check if at least 3 of them are the same
def check_line(line):
    if line.count(line[0]) == 4 and line[0] != 0:
        return True
    return False

print(check_line([1, 0, 0, 0]))
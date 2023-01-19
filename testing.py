class A:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b

    def __hash__(self):
        return hash(self.a) + hash(self.b)


d = {}

a1 = A(1, 2)
a2 = A(1, 3)

d[a1] = 2

print(d[a2])

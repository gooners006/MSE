class A:
    def top(self):
        print("top")


class B(A):
    def middle(self):
        print("middle")


class C(B):
    'Hello C'
    def bottom(self):
        print("bottom")

if __name__ == "__main__":
    obj=C()
    obj.top()
    obj.middle()
    obj.bottom()

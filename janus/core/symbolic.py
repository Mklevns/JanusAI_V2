# Expression: symbolic building block
class Expression:
    def __init__(self, expr):
        self.expr = expr

    def __repr__(self):
        return f"Expression({self.expr})"

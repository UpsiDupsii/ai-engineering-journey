import ast
import operator

CALCULATOR_DESCRIPTION = """
Tool Name: calculator
Description: Useful for when you need to perform mathematical calculations.
Action Input must be a valid mathematical expression in Python syntax (e.g., '2 + 2' or '250 * (4 / 2)').
"""

def execute_calculator(expression: str) -> str:
    """Safely evaluates basic mathematical expressions using an Abstract Syntax Tree (AST)."""
    allowed_operators = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg
    }

    def _eval(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return allowed_operators[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return allowed_operators[type(node.op)](_eval(node.operand))
        else:
            raise TypeError(f"Unsupported mathematical operation: {node}")

    try:
        node = ast.parse(expression, mode='eval').body
        result = _eval(node)
        return str(result)
    except Exception as e:
        return f"Error: Invalid expression. Details: {str(e)}"
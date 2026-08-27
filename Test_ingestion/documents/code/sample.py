"""Sample Python module for code-ingestion testing."""

def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number (0-indexed)."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

class BankAccount:
    """A minimal bank account with deposit/withdraw operations."""

    def __init__(self, owner: str, balance: float = 0.0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self.balance += amount

    def withdraw(self, amount: float) -> None:
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount

if __name__ == "__main__":
    print(f"Fib(10) = {fibonacci(10)}")
    acc = BankAccount("Shikher", 1000)
    acc.deposit(500)
    print(f"Balance: {acc.balance}")

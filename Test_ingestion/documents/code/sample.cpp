// Sample C++ file for code-ingestion testing
#include <iostream>
#include <vector>

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

class Rectangle {
public:
    double width, height;
    Rectangle(double w, double h) : width(w), height(h) {}
    double area() const { return width * height; }
};

int main() {
    std::cout << "Factorial of 5: " << factorial(5) << std::endl;
    Rectangle r(4.0, 5.0);
    std::cout << "Area: " << r.area() << std::endl;
    return 0;
}

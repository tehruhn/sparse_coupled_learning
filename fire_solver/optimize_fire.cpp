#include <iostream>
#include <Eigen/Dense>
#include <chrono>

// brew install eigen
// g++ -std=c++11 -o optimize_fire -I /opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/ optimize_fire.cpp

using namespace Eigen;

Vector2d gradf(Vector2d x, Vector2d params) {
    double a = params[0];
    double b = params[1];
    Vector2d grad;
    grad[0] = -2 * (a - x[0]) - 4 * b * (x[1] - x[0] * x[0]) * x[0];
    grad[1] = 2 * b * (x[1] - x[0] * x[0]);
    return grad;
}

double f(Vector2d x, Vector2d params) {
    double a = params[0];
    double b = params[1];
    return std::pow((a - x[0]), 2) + b * std::pow((x[1] - x[0] * x[0]), 2);
}

std::tuple<Vector2d, double, int> optimize_fire(Vector2d x0, Vector2d params, double atol = 1e-4, double dt = 0.002) {
    const double alpha0 = 0.1;
    const int Ndelay = 5;
    const int Nmax = 10000;
    const double finc = 1.1;
    const double fdec = 0.5;
    const double fa = 0.99;

    double error = 10 * atol;
    double dtmax = 10 * dt;
    double dtmin = 0.02 * dt;
    double alpha = alpha0;
    int Npos = 0;

    Vector2d x = x0;
    Vector2d V = Vector2d::Zero();
    Vector2d F = -gradf(x, params);
    double F_norm = F.norm();

    for (int i = 0; i < Nmax; ++i) {
        double P = F.dot(V);

        if (P > 0) {
            ++Npos;
            if (Npos > Ndelay) {
                dt = std::min(dt * finc, dtmax);
                alpha *= fa;
            }
        } else {
            Npos = 0;
            dt = std::max(dt * fdec, dtmin);
            alpha = alpha0;
            V.setZero();
        }

        V += 0.5 * dt * F;
        double V_norm = V.norm();
        V = (1 - alpha) * V + alpha * F * V_norm / F_norm;

        x += dt * V;
        F = -gradf(x, params);
        F_norm = F.norm();

        V += 0.5 * dt * F;
        error = F.array().abs().maxCoeff();

        if (error < atol) {
            return std::make_tuple(x, f(x, params), i);

        }
    }

    return std::make_tuple(x, f(x, params), Nmax);

}

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    Eigen::Vector2d x0(3.0, 4.0);
    Eigen::Vector2d params(1, 100);
    Eigen::Vector2d xmin;
    double fmin;
    int Niter;

    std::tie(xmin, fmin, Niter) = optimize_fire(x0, params);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;

    std::cout << "xmin = [" << xmin[0] << ", " << xmin[1] << "]" << std::endl;
    std::cout << "fmin = " << fmin << std::endl;
    std::cout << "Iterations = " << Niter << std::endl;

    return 0;
}
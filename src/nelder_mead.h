#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

double cost_function(const double* x){
    return pow((100 - x[0]), 2);
}

bool cmp(const Eigen::VectorXd& a, const Eigen::VectorXd& b){
    const double* a_array = a.data();
    const double* b_array = b.data();
    return cost_function(a_array) < cost_function(b_array);
}

class Nelder_Mead{
public:
    Nelder_Mead(double step=0.1, int max_iter=50, double alpha=1, double gamma=2, double rho=-0.5, double sigma=0.5): 
        step(step), max_iter(max_iter), alpha(alpha), gamma(gamma), rho(rho), sigma(sigma){}

    void solve(double* para, int dim){
        // Make polytope
        std::vector<Eigen::VectorXd> polytope;
        for(int i = 0; i < dim + 1; i++){
            Eigen::VectorXd vec = Eigen::Map<Eigen::VectorXd>(para, dim);
            if(i == dim){
                polytope.push_back(vec);
                continue;
            }
            vec(i) += step;
            polytope.push_back(vec);
        }        
        int iter = 0;
        while(1){
            if(iter >= max_iter) break;
            iter++;
            sort(polytope.begin(), polytope.end(), cmp);
            // Centroid
            Eigen::VectorXd x0(dim);
            x0.setZero();
            for(int i = 0; i < polytope.size() - 1; i++){
                for(int j = 0; j < polytope[i].size(); j++){
                    x0[j] += polytope[i](j);
                }
            }
            x0 /= (polytope.size() - 1);
            double current_best_score = cost_function(polytope[0].data());
            double worst_score = cost_function(polytope.back().data());
            double second_worst_score = cost_function(polytope[polytope.size() - 2].data());

            // Reflection
            Eigen::VectorXd xr = x0 + alpha * (x0 - polytope.back());
            double rscore = cost_function(xr.data());
            if(current_best_score <= rscore && rscore < second_worst_score){
                polytope.pop_back();
                polytope.push_back(xr);
                continue;
            }

            // Expansion
            if(rscore < current_best_score){
                Eigen::VectorXd xe = x0 + gamma * (x0 - polytope.back());
                double escore = cost_function(xe.data());
                if(escore < rscore){
                    polytope.pop_back();
                    polytope.push_back(xe);
                }
                else{
                    polytope.pop_back();
                    polytope.push_back(xr);
                }
                continue;
            }

            // Contraction
            Eigen::VectorXd xc = x0 + rho * (x0 - polytope.back());
            double cscore = cost_function(xc.data());
            if(cscore < worst_score){
                polytope.pop_back();
                polytope.push_back(xc);
                continue;
            }

            // Shrink
            Eigen::VectorXd x1 = polytope[0];
            for(int i = 1; i < polytope.size(); i++){
                polytope[i] = x1 + sigma * (polytope[i] - x1);
            }
        }
        std::cout << polytope[0] << std::endl;
    }
private:
    const double alpha, gamma, rho, sigma;
    const double step;
    const int max_iter;
};

from utilities.solutionFSHT import Solution
from utilities.fs_utils import transfer_function
import copy
import numpy as np
import sys

#faFSHT - firefly feature selection hyper-parameters tuning
#implementacija algoritma koja je prilagodjena i za fs i za hipertuning
class FA:
    def __init__(self, n, function, tf, nfeatures):
        
        self.N = n
        self.function = function
        self.population = []
        self.best_solution = [None] * self.function.D
        self.gamma = 1         # fixed light absorption coefficient
        self.beta_zero = 1         # attractiveness at distance zero
        self.beta = 1             
        self.alpha = 0.5         # randomization parameter
        self.FFE = self.N;  #parametar za fitness function evaluations, omdah se racuna za fazu inicijalizacije i jednak je broju resenja
        self.tf = tf; # ovo je transfer function za transformaciju u binary space
        self.nfeatures = nfeatures; #ovo je max broj features u datasetu, prvih nfeatures parametara su za features, ostali su za hiper-param optimiation


    def initial_population(self):
        for i in range(0, self.N):
            local_solution = Solution(self.function,self.tf,self.nfeatures)
            self.population.append(local_solution)
        self.population.sort(key=lambda x: x.objective_function)
        self.best_solution = copy.deepcopy(self.population[0].x)

    def update_position(self, t, max_iter):
        
        delta = 1 - (10**(-4)/0.9)**(1/max_iter)
        #self.alpha = ( 1- delta) * self.alpha
        
        lb = self.function.lb
        ub = self.function.ub
        
        scale = []
        for i in range(self.function.D):
            scale.append(np.abs(ub[i] - lb[i]))
        
        #temp_population = copy.deepcopy(self.population)
        temp_population = self.population.copy()
        
        for i in range(self.N):
          
            for j in range(self.N):

                if self.population[i].objective_function > temp_population[j].objective_function:
                    #print(self.nfeatures)
                    r = np.sqrt(np.sum((np.array(self.population[i].x) - (np.array(temp_population[j].x)))**2))                    
                    beta = (self.beta - self.beta_zero) * np.exp(-self.gamma * r ** 2) + self.beta_zero
                    temp = self.alpha * (np.random.rand(self.function.D) - 0.5) * scale
                    sol = np.array(self.population[i].x) * (1 - beta) + np.array(temp_population[j].x) * beta + temp

                    sol[0:self.nfeatures] = transfer_function(sol[0:self.nfeatures], self.tf, self.nfeatures)  #ovde koristimo transfer funckiju za prvih nfeatures parametara
                    sol = self.checkBoundaries(sol)

                    solution = Solution(self.function, self.tf,self.nfeatures,sol.tolist())
                    self.FFE = self.FFE + 1
             
                    if solution.objective_function < self.population[i].objective_function:
                        self.population[i] = solution
        #self.qr()
    def sort_population(self):

        self.population.sort(key=lambda x: x.objective_function)
        self.best_solution = self.population[0].x

    def get_global_best(self):
        return (self.population[0].objective_function, self.population[0].error,self.population[0].y_proba, self.population[0].y,self.population[0].feature_size,
                self.population[0].model)
        
        
        #return self.population[0].objective_function
    
    def get_global_worst(self):
        return self.population[-1].objective_function
    
    def optimum(self):
        print('f(x*) = ', self.function.minimum, 'at x* = ', self.function.solution)
        
    def algorithm(self):
        return 'FA'
    
    def objective(self):
        
        result = []
        
        for i in range(self.N):
            result.append(self.population[i].objective_function)
            
        return result
    
    def average_result(self):
        return np.mean(np.array(self.objective()))
    
    def std_result(self):        
        return np.std(np.array(self.objective()))
    
    def median_result(self):
        return np.median(np.array(self.objective()))
        
       
    def print_global_parameters(self):
            for i in range(0, len(self.best_solution)):
                 print('X: {}'.format(self.best_solution[i]))
                 
    def get_best_solutions(self):
        return np.array(self.best_solution)

    def get_solutions(self):
        
        sol = np.zeros((self.N, self.function.D))
        for i in range(len(self.population)):
            sol[i] = np.array(self.population[i].x)
        return sol


    def print_all_solutions(self):
        print("******all solutions objectives**********")
        for i in range(0,len(self.population)):
              print('solution {}'.format(i))
              print('objective:{}'.format(self.population[i].objective_function))
              print('solution {}: '.format(self.population[i].x))
              print('--------------------------------------')

    def get_global_best_params(self):
        return self.population[0].x

    #proverava boundaries za hiper-parametre
    def checkBoundaries(self,Xnew):
        for j in range(self.nfeatures,self.function.D):
            if Xnew[j] < self.function.lb[j]:
                Xnew[j] = self.function.lb[j]

            elif Xnew[j] > self.function.ub[j]:
                Xnew[j] = self.function.ub[j]
        return Xnew



    def getFFE(self):
        return self.FFE

    # funkcija koja vraca najbolji global_best_solution
    def get_global_best_solution(self):
        # ovde pravimo liste sa objective i indicator za celu populaciji

        indicator_list = []  # ovo je indikator, sta god da je u pitanju
        objective_list = []  # ovo je objective, sta god da je u pitanju
        objective_indicator_list = []
        for i in range(len(self.population)):
            indicator_list.append(
                self.population[i].error)  # ovo je za error, mada je to bilo koji drugi indikator, samo se tako zoeve
            objective_list.append(self.population[i].objective_function)  # ovo je objective
        objective_indicator_list.append(objective_list)
        objective_indicator_list.append(indicator_list)
        self.population[0].diversity = objective_indicator_list

        return self.population[0]




    def qr(self):
    #metoda za quasi-reflextive learning

        lb = self.function.lb
        ub = self.function.ub
        qr_solution = [None] * self.function.D
        for i in range(self.N):
            for j in range(self.function.D):
                if self.population[i].x[j] < (ub[j] + lb[j]) / 2:
                    qr_solution[j] = self.population[i].x[j] + (
                            (ub[j] + lb[j]) / 2 - self.population[i].x[j]) * np.random.uniform()
                else:
                    qr_solution[j] = (ub[j] + lb[j]) / 2 + (
                            self.population[i].x[j] - (ub[j] + lb[j]) / 2) * np.random.uniform()
            qr_solution_add = Solution(self.function, qr_solution)
            self.population.append(qr_solution_add)
        self.population.sort(key=lambda x: x.objective_function)
        # delete elements from population that are not needed
        del self.population[(self.N):len(self.population)]
        # self.FFE = self.FFE + self.N  # dodajemo FFE za QRBL mechanism






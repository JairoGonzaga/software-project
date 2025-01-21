from utils.ssl.Navigation import Navigation, Point
from utils.ssl.base_agent import BaseAgent
import numpy as np

class ExampleAgent(BaseAgent):
    def __init__(self, id=0, yellow=False):
        super().__init__(id, yellow)

    def dwa_navigation(self, goal, obstacles, params):

        pos = [self.robot.x, self.robot.y, self.body_angle]  # Posição atual do robô
        max_distance = params['max_distance']  # Distância máxima a explorar
        step_size = params['step_size']  # Incremento para simulação de pontos
        safe_margin = params['safe_distance']  # Margem segura de obstáculos

        best_score = float('-inf')  # Melhor pontuação encontrada
        best_next_point = Point(pos[0], pos[1])  # Melhor ponto de destino

        # Simula vários pontos ao redor do robô
        for d in np.arange(step_size, max_distance + step_size, step_size):  # Distâncias incrementais
            for angle in np.linspace(-np.pi, np.pi, params['num_directions']):  # Ângulos 360°

                # Calcula o próximo ponto
                x_next = pos[0] + d * np.cos(angle)
                y_next = pos[1] + d * np.sin(angle)

                # Avalia distância ao objetivo
                dist_to_goal = -np.linalg.norm([x_next - goal[0], y_next - goal[1]])

                # Avalia distância mínima a obstáculos
                safe_distance = min(
                    np.linalg.norm([x_next - obs[0], y_next - obs[1]]) for obs in obstacles
                )
               
                # Penaliza pontos que estejam muito próximos de obstáculos
                if safe_distance < safe_margin:
                        score = dist_to_goal / safe_distance
                else:
                    # Incentiva pontos seguros
                    score = dist_to_goal + safe_distance*0.5
                

                # Atualiza o melhor ponto
                if score > best_score:
                    best_score = score
                    best_next_point = Point(x_next, y_next)
        return best_next_point



    def decision(self): # funcao que decide o que o robo vai fazer
        if len(self.targets) == 0: # se nao tiver alvo, ele nao faz nada
            return
        target_assigments = {target: None for target in self.targets}
        for target in self.targets:
            closest_teammate_id = None
            closest_distance = float('inf')

            for teammate_id, teammate in self.teammates.items():
                distance = np.linalg.norm([target.x - teammate.x, target.y - teammate.y])
                if distance < closest_distance:
                    closest_distance = distance
                    closest_teammate_id = teammate_id
                        
            target_assigments[target] = closest_teammate_id

        for target, assigned_id in target_assigments.items():
            if assigned_id == self.id:
                goal = target
                obstacles = [[obs.x, obs.y] for obs in self.opponents.values()] #lista de obstaculos

                distanciasegura = 0.5 # distancia usada para poder usar ou nao oDWA (Dynamic Window Approach) evitando que ele fique muito lento ja que nao precisa rodar sempre
                obistaculo_proximo = any(np.linalg.norm([self.robot.x - obs.x, self.robot.y - obs.y]) < distanciasegura for obs in self.opponents.values()) 
                # se tiver um obstaculo proximo ele roda o DWA
                proximogol = np.linalg.norm([self.robot.x - goal[0], self.robot.y - goal[1]]) < 0.5 
                print(proximogol)
                if proximogol:
                    next_point = Point(goal[0], goal[1])
                else:
                    if obistaculo_proximo:
                        # Parâmetros do DWA
                        params = {
                            'max_distance': 0.31,  # Distância máxima a simular
                            'step_size': 0.2,  # Incremento de distância para pontos simulados
                            'num_directions': 36,  # Número de direções a considerar (360° dividido uniformemente)
                            'safe_distance': 0.4,  # Distância mínima segura de obstáculos
                            'goal_tolerance': 0.4   # Distância para considerar que está suficientemente próximo do objetivo
                        }

                    

                        # Calcula o próximo ponto seguro com o DWA
                        next_point = self.dwa_navigation(goal, obstacles, params)
                    else:
                        next_point = Point(goal[0], goal[1]) #

                if next_point:
                    next_point = Point(next_point[0], next_point[1])  # Converte para objeto Point
                    # Envia o próximo ponto para o Navigation para controlar o robô
                    target_velocity, target_angle_velocity = Navigation.goToPoint(self.robot, next_point)
                    self.set_vel(target_velocity)
                    self.set_angle_vel(target_angle_velocity)

    def parado(self):
        self.set_vel(Point(0, 0))
        self.set_angle_vel(0)
    
    def post_decision(self):
        pass

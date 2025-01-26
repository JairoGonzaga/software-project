from utils.ssl.Navigation import Navigation, Point
from utils.ssl.base_agent import BaseAgent
from decisao import atribuir_alvos
from dwa import dwa_navigation
import numpy as np


class ExampleAgent(BaseAgent):
    def __init__(self, id=0, yellow=False):
        super().__init__(id, yellow)
        self.contador_objparado = 0  # Inicializa o contador de evitamento de obstáculos
        self.position_history = [] # Inicializa o histórico de posições
        self.stuck_time = 2  # Tempo em segundos para considerar que está preso

    def Updt_pos(self, position):#armazena a posição atual do robô
        self.position_history.append(position)
        if len(self.position_history) > self.stuck_time * 10: 
            self.position_history.pop(0)
        
    def verificar_obstaculos_proximos(self, goal, obstacles):#função auxiliar para verificar obstáculos próximos, usada em decidir prox ponto
        distanciasegura = 0.565743  # Distância usada para considerar um obstáculo próximo
        obistaculo_proximo = any(np.linalg.norm([self.robot.x - obs[0], self.robot.y - obs[1]]) < distanciasegura for obs in obstacles) # se o obstáculo estiver próximo do robo
        proximogol = np.linalg.norm([self.robot.x - goal[0], self.robot.y - goal[1]]) < 0.32 # se o robô estiver próximo do objetivo
        obstaculo_muito_proximo = any(np.linalg.norm([goal[0] - obs[0], goal[1] - obs[1]]) < 0.185 for obs in obstacles) and proximogol # se o obstáculo estiver muito próximo do objetivo

        return obistaculo_proximo, proximogol, obstaculo_muito_proximo

    def decidir_proximo_ponto(self, goal, obstacles):
        # Atualiza a posição atual do robô
        current_position = [self.robot.x, self.robot.y]
        self.Updt_pos(current_position)

        obistaculo_proximo, proximogol, obstaculo_muito_proximo = self.verificar_obstaculos_proximos(goal, obstacles) #valores auxiliares para decidir o próximo ponto
        if proximogol:# se o robô estiver próximo do objetivo, vá diretamente para o objetivo
            return Point(goal[0], goal[1])
        else:
            if obistaculo_proximo:
                # Parâmetros do DWA
                params = {
                    'max_distance': 0.2891323789,  # Distância máxima a simular
                    'step_size': 0.243,  # Incremento de distância para pontos simulados
                    'num_directions': 20,  # Número de angulos a considerar (360° dividido uniformemente)
                    'safe_distance': 0.282717,  # Distância mínima segura de obstáculos
                }

                # Se o contador exceder o limite, vá diretamente para o objetivo
                if obstaculo_muito_proximo: #isso é uma tentativa de evitar que o robô fique preso em um loop de evitamento de obstáculos
                    return Point(goal[0], goal[1])
                else:
                    # Calcula o próximo ponto seguro com o DWA
                    return dwa_navigation(self, goal, obstacles, params) 
            else:# se tiver livre, ele vai ao ponto
                return Point(goal[0], goal[1])

    def decision(self):# função de gerenciamento geral do robo
        """
        Função que decide o que o robô vai fazer.
        """
        if len(self.targets) == 0:  # Se não tiver alvo, ele não faz nada
            return

        # Atribui alvos aos robôs mais próximos
        target_assigments = atribuir_alvos(self.targets, self.teammates)

        # Verifica se o robô tem um alvo atribuído e manda o objetivo para ele
        if self.id in target_assigments.values():
            goal = [target for target, assigned_id in target_assigments.items() if assigned_id == self.id][0]
            goal = [goal.x, goal.y]
            obstacles = [[obs.x, obs.y] for obs in self.opponents.values()]

            next_point = self.decidir_proximo_ponto(goal, obstacles)# decide o proximo ponto para o robô se mover

            if next_point:
                # Envia o próximo ponto para o Navigation para controlar o robô
                target_velocity, target_angle_velocity = Navigation.goToPoint(self.robot, next_point)
                self.set_vel(target_velocity)
                self.set_angle_vel(target_angle_velocity)

    def post_decision(self):
        pass

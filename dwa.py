from utils.ssl.Navigation import Navigation, Point
from utils.ssl.base_agent import BaseAgent
import pygame as pygame
import numpy as np

def dwa_navigation(self, goal, obstacles, params):
    pos = [self.robot.x, self.robot.y, self.body_angle]  # Posição atual do robô
    max_distance = params['max_distance']  # Distância máxima a simular
    step_size = params['step_size']  # Incremento de distância para pontos simulados
    safe_dist = params['safe_distance']  # Distância mínima segura de obstáculos
    ta_preso = 0.3  # Distância para considerar que está preso

    best_score = float('-inf')
    best_next_point = Point(pos[0], pos[1])

    # Simula vários pontos ao redor do robô
    for d in np.arange(step_size, max_distance + step_size, step_size):  # Distâncias incrementais
        for angle in np.linspace(-np.pi, np.pi, params['num_directions']):  # Ângulos 360°

            x_next = pos[0] + d * np.cos(angle)
            y_next = pos[1] + d * np.sin(angle)

            # Avalia distância ao objetivo
            dist_to_goal = np.linalg.norm([x_next - goal[0], y_next - goal[1]])
            # Avalia distância mínima a obstáculos
            safe_distance = min(
                np.linalg.norm([x_next - obs[0], y_next - obs[1]]) for obs in obstacles
            )

            # Penaliza pontos que estejam muito próximos de obstáculos
            if safe_distance < safe_dist:
                score = -dist_to_goal - 1 / safe_distance  # Penalização inversa da distância
            else:
                score = -dist_to_goal + safe_distance  # Incentiva pontos seguros

            # Penaliza pontos que estão muito próximos das posições anteriores
            if hasattr(self, 'position_history'):
                recent_positions = self.position_history[-20:]  # Últimas 10 posições
                for pos in recent_positions:
                    if np.linalg.norm([x_next - pos[0], y_next - pos[1]]) < ta_preso:
                        score -= 20  # Penalização adicional

            # Atualiza o melhor ponto
            if score > best_score:
                best_score = score
                best_next_point = Point(x_next, y_next)
                

    return best_next_point
import numpy as np

def atribuir_alvos(targets, teammates):
    """
    Atribui alvos aos robôs mais próximos.
    """
    target_assigments = {target: None for target in targets} #inicialização
    for target in targets: 
        closest_teammate_id = None
        closest_distance = float('inf')

        for teammate_id, teammate in teammates.items():#para cada robô, verifica a distância entre ele e o alvo
            distance = np.linalg.norm([target.x - teammate.x, target.y - teammate.y])
            if distance < closest_distance:
                closest_distance = distance
                closest_teammate_id = teammate_id

        target_assigments[target] = closest_teammate_id#atribui o alvo ao robô mais próximo

    return target_assigments #retorna o dicionário com os alvos atribuídos
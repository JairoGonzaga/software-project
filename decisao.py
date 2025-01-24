import numpy as np

def atribuir_alvos(targets, teammates):
    targets_array = np.array([[target.x, target.y] for target in targets])
    teammates_array = np.array([[teammate.x, teammate.y] for teammate in teammates.values()])
    teammate_ids = list(teammates.keys())
    
    # Calcula distâncias em lote
    dists = np.linalg.norm(targets_array[:, None, :] - teammates_array[None, :, :], axis=2)
    
    # Atribui cada alvo ao robô mais próximo
    target_assigments = {}
    for i, target in enumerate(targets):
        closest_idx = np.argmin(dists[i])
        target_assigments[target] = teammate_ids[closest_idx]
    
    return target_assigments

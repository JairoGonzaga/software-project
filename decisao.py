import numpy as np

def atribuir_alvos(targets, teammates):#função que atribui alvos aos robôs mais próximos
    alvo = np.array([[target.x, target.y] for target in targets])
    time = np.array([[teammate.x, teammate.y] for teammate in teammates.values()])
    time_id = list(teammates.keys())
    
    # Calcula distâncias em lote
    dists = np.linalg.norm(alvo[:, None, :] - time[None, :, :], axis=2)
    
    # Atribui cada alvo ao robô mais próximo
    Alvos_pegos = {}
    for i, target in enumerate(targets):
        closest_idx = np.argmin(dists[i])
        Alvos_pegos[target] = time_id[closest_idx]
    
    return Alvos_pegos

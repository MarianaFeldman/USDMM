import WOMC_V2_parallel

if __name__ == "__main__":

    WOMC = WOMC_V2_parallel.WOMC(
    new = True, # True/False -> inicializar aleatório ou inicializar de um arquivo
    nlayer = 2, # INT -> número de W-operadores em camada
    wlen = 3,  # INT -> tamnho do W-operador (wlen*wlen)
    train_size = 30, # INT -> qtd de imagens de treinamento 
    val_size = 10, # INT -> qtd de imagens de validação
    test_size = 10, # INT -> qtd de imagens de teste
    error_type = 'iou', # 'mae' / 'iou' -> tipo de erro no cálculo
    neighbors_sample = 10, # INT/False -> se vamos amostrar vizinhos e se sim quantos
    epoch_f = 100, # INT -> quantidade de épocas no reticulado das funções booleanas (janela fixa)
    epoch_w = 20, # INT -> quantidade de épocas no reticulado das janelas
    batch = 1, # INT -> qtd de imagens de treino utilizada
    path_results = 'results_V6', # STR -> pasta para ser salvo os resultados das imagens
    name_save='_V6', # STR -> final a ser salvo nos arquivos (W, joint, W_hist)
    seed = 0, #INT -> seed para aleatórios (garantir reproducibilidade)
    parallel = True # True/False -> utilizar função paralela (True) ou sequencial (False)
    )

    WOMC.fit()

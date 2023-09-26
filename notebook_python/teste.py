import WOMC_V2

if __name__ == "__main__":

    WOMC = WOMC_V2.WOMC(
    new = True, # True/False -> inicializar aleatório ou inicializar de um arquivo
    nlayer = 2, # INT -> número de W-operadores em camada
    wlen = 3,  # INT -> tamnho do W-operador (wlen*wlen)
    train_size = 30, # INT -> qtd de imagens de treinamento 
    val_size = 10, # INT -> qtd de imagens de validação
    test_size = 10, # INT -> qtd de imagens de teste
    error_type = 'iou', # 'mae' / 'iou' -> tipo de erro no cálculo
    neighbors_sample = 10, # INT/False -> se vamos amostrar vizinhos e se sim quantos
    epoch_f = 500, # INT -> quantidade de épocas no reticulado das funções booleanas (janela fixa)
    epoch_w = 50, # INT -> quantidade de épocas no reticulado das janelas
    batch = 1, # INT -> qtd de imagens de treino utilizada
    path_results = 'results_V4', # STR -> pasta para ser salvo os resultados das imagens
    name_save='_V4', # STR -> final a ser salvo nos arquivos (W, joint, W_hist)
    seed = 0, #INT -> seed para aleatórios (garantir reprodutibilidade )
    parallel = True, # True/False -> utilizar função paralela (True) ou sequencial (False)
    early_stop_round_f = 50, #INT -> qtd de épocas máximas no reticulado das funções booleanas sem decréscimo no erro
    early_stop_round_w = 10 #INT -> qtd de épocas máximas no reticulado das janelas sem decréscimo no erro
    )

    WOMC.fit()
    #WOMC.teste(True)

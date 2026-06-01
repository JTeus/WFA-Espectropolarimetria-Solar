# Fundamentos de Espectropolarimetria Solar (WFA)

Este repositório contém os cadernos interativos (*Jupyter Notebooks*) e o módulo de funções analíticas em Python desenvolvidos para o artigo científico:
**"Fundamentos de Espectropolarimetria Solar: Uma abordagem clássica via aproximação de campo fraco"**.

## Autor
**José Matheus da Silva Rocha** Instituto Nacional de Pesquisas Espaciais (INPE), São José dos Campos, SP, Brasil.

## Estrutura do Repositório
- `solar_utils.py`: Módulo contendo a física central do projeto (leitura FITS, perfis de Voigt/Zeeman, derivadas numéricas e regressões lineares WFA).
- `Data_PREP_paper.ipynb`: Calibração espectral, visualização espacial e validação do ajuste WFA para píxeis fotossféricos isolados.
- `Weak_Field_Aplication_paper.ipynb`: Inversão vetorial completa do mapa espectropolarimétrico para geração dos magnetogramas ($B_{LOS}$, $B_T$ e $\chi$).
- `Eventos/`: Diretório base configurado para leitura dos arquivos originais.
- `Figuras/`: Diretório de saída configurado para o salvamento automático dos gráficos gerados.

## Dependências
Para executar os códigos, certifique-se de instalar as seguintes bibliotecas:
```bash
pip install numpy matplotlib astropy

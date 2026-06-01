# solar_utils.py
import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

# DICIONÁRIO DE LINHAS ESPECTRAIS
SPECTRAL_LINES = {
    '6301': {
        'lambda_0': 6301.5,
        'g_eff': 1.667, 
        'G_bar': 2.517 
    },
    '6302': {
        'lambda_0': 6302.5,
        'g_eff': 2.500, 
        'G_bar': 6.250 
    }
}

# FUNÇÕES DE LEITURA E CALIBRAÇÃO
def load_solar_fits(filepath):
    """
    Carrega o arquivo FITS e extrai os dados do cabeçalho e os arrays principais.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"O arquivo {filepath} não foi encontrado.")
    with fits.open(filepath) as hdul:
        data = hdul[0].data.astype("float64")
        header = hdul[0].header
    return data, header

def generate_wavelength_axis(stokes_data, header):
    """
    Gera o eixo de comprimentos de onda usando os dados de calibração do cabeçalho FITS.
    Utiliza as chaves CRPIX1, CRVAL1 e CDELT1.
    """
    NL = stokes_data.shape[3]
    ll_idx = np.linspace(0, NL - 1, NL)

    # Extrai do cabeçalho ou usa os valores padrão se a chave não existir
    crpix1 = header.get('CRPIX1', 56.5000)
    crval1 = header.get('CRVAL1', 6302.08)
    cdelt1 = header.get('CDELT1', 0.0215) 

    # Equação de calibração real
    ll = crval1 + (ll_idx - crpix1) * cdelt1
    return ll

def normalize_stokes_continuum(stokes_data, num_points=10):
    """
    Calcula a intensidade do contínuo (Sol Calmo) e normaliza todo o cubo de dados.
    """
    I_continuum = np.mean(stokes_data[:, :, 0, :num_points], axis=-1)
    I_qs = np.max(I_continuum)
    stokes_norm = stokes_data / I_qs
    mapa_intensidade_norm = I_continuum / I_qs
    return stokes_norm, I_qs, mapa_intensidade_norm

# FUNÇÕES DE COORDENADAS E ESCALAS FÍSICAS
def get_physical_extent(header, nx, ny, default_xscale=0.29714, default_yscale=0.31998):
    """
    Lê o cabeçalho FITS e converte as dimensões em pixels para Arcsec.
    """
    XSCALE = header.get('XSCALE', default_xscale)
    YSCALE = header.get('YSCALE', default_yscale)
    XCEN = header.get('XCEN', 0.0)
    YCEN = header.get('YCEN', 0.0)

    largura_x = nx * XSCALE
    altura_y = ny * YSCALE

    esquerda = XCEN - (largura_x / 2.0)
    direita  = XCEN + (largura_x / 2.0)
    baixo    = YCEN - (altura_y / 2.0)
    cima     = YCEN + (altura_y / 2.0)

    return [esquerda, direita, baixo, cima]

def get_physical_scales(shape_x, shape_y, header_params, unidade='arcsec'):
    """
    Calcula as escalas físicas, o extent do mapa e as strings de unidade.
    """
    x_scale_arc = header_params.get('XSCALE', 0.29714)
    y_scale_arc = header_params.get('YSCALE', 0.31998)
    xcen_arc = header_params.get('XCEN', 336.664)
    ycen_arc = header_params.get('YCEN', -394.824)
    km_per_arc = 720.0

    if unidade.lower() == 'km':
        x_scale, y_scale = x_scale_arc * km_per_arc, y_scale_arc * km_per_arc
        xcen, ycen = xcen_arc * km_per_arc, ycen_arc * km_per_arc
        unidade_str = 'km'
    elif unidade.lower() == 'arcsec':
        x_scale, y_scale = x_scale_arc, y_scale_arc
        xcen, ycen = xcen_arc, ycen_arc
        unidade_str = 'arcsec'
    else:
        raise ValueError("UNIDADE_DESEJADA deve ser 'km' ou 'arcsec'")

    largura_x, altura_y = shape_x * x_scale, shape_y * y_scale
    esquerda, direita = xcen - (largura_x / 2.0), xcen + (largura_x / 2.0)
    baixo, cima = ycen - (altura_y / 2.0), ycen + (altura_y / 2.0)

    return x_scale, y_scale, [esquerda, direita, baixo, cima], unidade_str

def physical_to_pixel(posicao_fisica, limite_inferior, escala_pixel):
    px_decimal = (posicao_fisica - limite_inferior) / escala_pixel
    return int(round(px_decimal))

# APROXIMAÇÃO DE CAMPO FRACO (WFA)
def vectorized_estimate_wfa(I, Q, U, V, ll, lam, g, G_bar):
    """
    Executa a Aproximação de Campo Fraco (WFA) de forma vetorizada 
    (para a imagem toda simultaneamente).
    """
    dll = np.gradient(ll)
    dI_dl = np.gradient(I, axis=2) / dll
    d2I_dl2 = np.gradient(dI_dl, axis=2) / dll
    
    n = I.shape[2] 
    k = 4.6686e-13  
    
    # Cálculo de Blos (Regressão Linear Vetorizada)
    x_los, y_los = dI_dl, V
    sum_x, sum_y = np.sum(x_los, axis=2), np.sum(y_los, axis=2)
    sum_xy, sum_xx = np.sum(x_los * y_los, axis=2), np.sum(x_los**2, axis=2)
    
    den_blos = (n * sum_xx - sum_x**2)
    den_blos[den_blos == 0] = 1e-10 
    m_blos = (n * sum_xy - sum_x * sum_y) / den_blos
    Blos = -m_blos / (k * (lam**2) * g)
    
    # Cálculo de Bt
    L = np.sqrt(Q**2 + U**2)
    x_t, y_t = np.abs(d2I_dl2), L
    sum_xt, sum_yt = np.sum(x_t, axis=2), np.sum(y_t, axis=2)
    sum_xyt, sum_xxt = np.sum(x_t * y_t, axis=2), np.sum(x_t**2, axis=2)
    
    den_bt = (n * sum_xxt - sum_xt**2)
    den_bt[den_bt == 0] = 1e-10
    m_bt = (n * sum_xyt - sum_xt * sum_yt) / den_bt
    Bt = np.sqrt(np.abs(m_bt * 4 / (G_bar * (lam**2 * k)**2)))
    
    # Cálculo do Azimute (Chi)
    idx_max_L = np.argmax(L, axis=2)
    NX, NY = L.shape[0], L.shape[1]
    ix, iy = np.indices((NX, NY))
    
    Q_max, U_max = Q[ix, iy, idx_max_L], U[ix, iy, idx_max_L]
    chi = 0.5 * np.arctan2(U_max, Q_max)
    chi_final = (np.degrees(chi) + 90) % 180
    
    return Blos, Bt, chi_final

def run_wfa_pipeline(stokes_data, ll, linha_escolhida='6301', tamanho_janela=14):
    """
    Função principal de automação. Prepara o recorte da janela espectral
    e dispara o cálculo vetorizado da WFA sem a necessidade de loops for.
    """
    params = SPECTRAL_LINES[linha_escolhida]
    lam, g, G_bar = params['lambda_0'], params['g_eff'], params['G_bar']
    
    print(f"Analisando a linha {lam} Å | g_eff = {g} | G_bar = {G_bar}")
    
    i_centro = np.argmin(np.abs(ll - lam))
    limite_esq = max(0, i_centro - tamanho_janela)
    limite_dir = min(len(ll), i_centro + tamanho_janela + 1)
    
    # Fatiamento dinâmico para a imagem toda
    I = stokes_data[:, :, 0, limite_esq : limite_dir]
    Q = stokes_data[:, :, 1, limite_esq : limite_dir]
    U = stokes_data[:, :, 2, limite_esq : limite_dir]
    V = stokes_data[:, :, 3, limite_esq : limite_dir]
    l_window = ll[limite_esq : limite_dir]
    
    return vectorized_estimate_wfa(I, Q, U, V, l_window, lam, g, G_bar)

# FUNÇÕES DE PLOTAGEM
def plot_magnetic_maps(Blos, Bt, Chi, map_extent, save_fig=False, filename='magnetogramas.png'):
    """
    Plota os mapas de Blos, Bt e Azimute lado a lado com zoom específico, 
    círculos delimitadores (Umbra/Penumbra) e formatação ajustada para publicação.
    """
    # Define o ponto central em zero para o Blos (divergente)
    norm = colors.TwoSlopeNorm(vmin=-4500, vcenter=0, vmax=Blos.max()) 

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 11))

    # Limites de zoom desejados
    zoom_x = [320, 380]
    zoom_y = [-440, -350]

    # Configuração dos círculos de região de interesse
    x_cen, y_cen = 349, -396
    regioes = [
        {'raio': 7,  'cor': 'g',  'label': 'Umbra'},
        {'raio': 16, 'cor': 'b', 'label': 'Penumbra'}
    ]

    # Aplica recortes e desenhos para todos os 3 eixos
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(zoom_x)
        ax.set_ylim(zoom_y)
        for r in regioes:
            circulo = patches.Circle(
                (x_cen, y_cen), r['raio'], edgecolor=r['cor'], 
                facecolor='none', linestyle='-', linewidth=2
            )
            ax.add_patch(circulo)

    # --- GRÁFICO 1: Blos ---
    im1 = ax1.imshow(Blos, origin='lower', extent=map_extent, cmap='RdBu_r', norm=norm)
    ax1.set_title("Campo Longitudinal ($B_{LOS}$)", fontsize=18)
    ax1.set_xlabel('X [arcsec]', fontsize=14)
    ax1.set_ylabel('Y [arcsec]', fontsize=14)
    ax1.tick_params(axis='both', labelsize=12)

    cbar1 = fig.colorbar(im1, ax=ax1, shrink=1, pad=0.01)
    cbar1.set_label('Gauss', fontsize=14)
    cbar1.ax.tick_params(labelsize=12)
    cbar1.locator = ticker.MultipleLocator(500)
    cbar1.update_ticks()

    # --- GRÁFICO 2: Bt ---
    im2 = ax2.imshow(Bt, origin='lower', extent=map_extent, cmap='Greys')
    ax2.set_title("Campo Transversal ($B_{T}$)", fontsize=18)
    ax2.set_xlabel('X [arcsec]', fontsize=14)
    ax2.tick_params(axis='both', labelsize=12)
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=1, pad=0.01)
    cbar2.set_label('Gauss', fontsize=14)
    cbar2.ax.tick_params(labelsize=12)

    # --- GRÁFICO 3: Azimute ---
    im3 = ax3.imshow(Chi, origin='lower', extent=map_extent, cmap='twilight', vmin=0, vmax=180)
    ax3.set_title(r"Azimute ($\chi$)", fontsize=18)
    ax3.set_xlabel('X [arcsec]', fontsize=14)
    ax3.tick_params(axis='both', labelsize=12)
    cbar3 = fig.colorbar(im3, ax=ax3, shrink=1, pad=0.01)
    cbar3.set_label('Graus', fontsize=14)
    cbar3.ax.tick_params(labelsize=12)
    cbar3.set_ticks([0, 45, 90, 135, 180])

    if save_fig:
        plt.savefig(filename, dpi=600, transparent=True, bbox_inches='tight', pad_inches=0.1)
    
    return fig

# FUNÇÕES EXPLORATÓRIAS (Data_PREP)
def plot_stokes_profiles_pixel(ll, stokes_norm, idx_x, idx_y, janela_limites=None, save_fig=False, filename='Stokes_IQUV_POINTS.png'):
    """
    Plota o perfil 1D dos quatro parâmetros de Stokes para um píxel específico.
    Permite sombrear a região de análise (janela WFA).
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    
    perfis = [
        (stokes_norm[idx_y, idx_x, 0, :], 'Stokes I/Ic', axs[0, 0], 'black'),
        (stokes_norm[idx_y, idx_x, 1, :], 'Stokes Q/Ic', axs[0, 1], 'blue'),
        (stokes_norm[idx_y, idx_x, 2, :], 'Stokes U/Ic', axs[1, 0], 'green'),
        (stokes_norm[idx_y, idx_x, 3, :], 'Stokes V/Ic', axs[1, 1], 'red')
    ]
    
    for perfil, titulo, ax, cor in perfis:
        ax.plot(ll, perfil, color=cor, linewidth=2)
        ax.set_title(titulo, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Sombreamento da janela espectral WFA
        if janela_limites is not None:
            ax.axvspan(ll[janela_limites[0]], ll[janela_limites[1]], color='gray', alpha=0.3)
            
        if ax in [axs[1, 0], axs[1, 1]]:
            ax.set_xlabel(r'Comprimento de Onda ($\AA$)', fontsize=12)

    plt.tight_layout()
    if save_fig:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    return fig

def plot_stokes_grid_validation(stokes_norm, mapa_cont, ll, map_extent, x_scale, y_scale, 
                                x_cen_arcsec, y_cen_arcsec, pontos_arcsec, 
                                save_fig=False, filename='validacao_stokes_grid.png'):
    """
    Gera o painel de validação com um mapa de zoom do contínuo em arcsec
    e uma grade 3x4 mostrando os 4 perfis de Stokes para 3 pontos escolhidos.
    
    Todos os parâmetros de posicionamento de entrada devem ser fornecidos em arcsec.
    """
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    fig = plt.figure(figsize=[20, 10], dpi=150)

    colors = ['r', 'g', 'b']
    labels = ['Ponto 1', 'Ponto 2', 'Ponto 3']

    # Layout principal
    gs_main = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.5], wspace=0.25)

    # PAINEL ESQUERDO: Mapa com Zoom (Escala Física)
    ax_map = fig.add_subplot(gs_main[0, 0])
    
    im = ax_map.imshow(mapa_cont, origin='lower', cmap='gray', 
                       extent=map_extent, vmin=0, vmax=1) 

    # Limites físicos da borda do mapa (origem)
    limite_esq_x = map_extent[0]
    limite_baixo_y = map_extent[2]

    # Converte o zoom antigo de 100 pixels para o equivalente exato em arcsec
    zoom_range_x = 100 * x_scale
    zoom_range_y = 100 * y_scale

    ax_map.set_xlim(x_cen_arcsec - zoom_range_x, x_cen_arcsec + zoom_range_x)
    ax_map.set_ylim(y_cen_arcsec - zoom_range_y, y_cen_arcsec + zoom_range_y)

    # Plota o marcadores diretamente com as coordenadas em arcsec
    for i, (pt_x_arcsec, pt_y_arcsec) in enumerate(pontos_arcsec):
        ax_map.plot(pt_x_arcsec, pt_y_arcsec, 'X', color=colors[i], ms=10, label=labels[i])

    ax_map.set_title("Região de Zoom (Intensidade)", fontsize=16)
    ax_map.set_xlabel("X [arcsec]", fontsize=15)
    ax_map.set_ylabel("Y [arcsec]", fontsize=15)
    ax_map.tick_params(axis='both', labelsize=12)
    ax_map.legend(loc='lower left')

    divider = make_axes_locatable(ax_map)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r'$I/I_c$', fontsize=15)

    # PAINEL DIREITO: A Grade 3x4 de Perfis
    gs_stokes = gs_main[0, 1].subgridspec(4, 3, hspace=0.07, wspace=0.02)
    axes_grid = gs_stokes.subplots(sharex=True, sharey='row')
    
    stokes_titles = ["Stokes $I/I_c$", "Stokes $Q/I_c$", "Stokes $U/I_c$", "Stokes $V/I_c$"]

    for i in range(4):     # 4 Parâmetros de Stokes (Linhas do Grid)
        for j in range(3): # 3 Pontos (Colunas do Grid)
            ax = axes_grid[i, j] 
            
            # Pega as coordenadas físicas em arcsec
            pt_x_arcsec, pt_y_arcsec = pontos_arcsec[j] 
            
            # Converte arcsec para índices inteiros de matriz (pixels) de forma robusta
            idx_x = physical_to_pixel(pt_x_arcsec, limite_esq_x, x_scale)
            idx_y = physical_to_pixel(pt_y_arcsec, limite_baixo_y, y_scale)
            
            # Extrai o perfil mantendo a sua ordenação original [Y, X, Parâmetro, Lambda]
            profile = stokes_norm[idx_y, idx_x, i, :]
            
            ax.plot(ll, profile, color=colors[j], lw=1.5)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Formatação de títulos e rótulos
            if i == 0:
                ax.set_title(labels[j], fontsize=14, color=colors[j], fontweight='bold')
            if j == 0:
                ax.set_ylabel(stokes_titles[i], fontsize=14, fontweight='bold')
            if i == 3:
                ax.set_xlabel(r"$\lambda$ ($\AA$)", fontsize=14)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Definição estrita das escalas dos eixos Y para publicação
    for j in range(3):
        axes_grid[0, j].set_ylim(0.0, 1.05)
        axes_grid[1, j].set_ylim(-0.15, 0.15)
        axes_grid[2, j].set_ylim(-0.15, 0.15)
        axes_grid[3, j].set_ylim(-0.25, 0.25)

    # Remove os rótulos do eixo Y das colunas internas para evitar poluição visual
    for i in range(4):
        for j in range(1, 3): 
            axes_grid[i, j].tick_params(axis='y', length=0)

    if save_fig:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    return fig

# 

def plot_overview_fenda_espectros(mapa_cont, stokes_norm, map_extent, x_scale, 
                                  x_cen=349, y_cen=-396, pos_fenda1=-397, pos_fenda2=-407, 
                                  raio_umbra=7, raio_penumbra=16, # <-- NOVOS PARÂMETROS AQUI
                                  zoom_x=(320, 380), zoom_y=(-440, -350),
                                  dispersao=0.0215, save_fig=False, filename='perfil_stokes_zeeman_versao2.png'):
    """
    Cria o mapa multipainel avançado com GridSpec: Mapa do contínuo à esquerda 
    (com zoom, setas e círculos) e os espectros de Stokes à direita com Régua Zeeman.
    """
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as patches
    
    # Descompacta limites físicos e obtém o índice do pixel da fenda
    esquerda, direita, baixo, cima = map_extent
    extent_espectral = [0, stokes_norm.shape[3], baixo, cima]
    
    # Assumindo que a função physical_to_pixel está definida no mesmo módulo
    idx_x_fenda = physical_to_pixel(x_cen, esquerda, x_scale)

    # CONFIGURAÇÃO DE FIGURA E GRID
    fig = plt.figure(figsize=[24, 11])
    gs = GridSpec(2, 8, height_ratios=[0.05, 1], width_ratios=[0.35, 0.35, 0.35, 0.35, 1, 1, 1, 1], figure=fig)

    cax = fig.add_subplot(gs[0, 1:3])
    caxs_stokes = [fig.add_subplot(gs[0, 4+i]) for i in range(4)]

    ax_map = fig.add_subplot(gs[1, 0:4])
    axs_stokes = [fig.add_subplot(gs[1, 4+i], sharey=ax_map) for i in range(4)]

    # PAINEL DA ESQUERDA: MAPA DO CONTÍNUO
    im = ax_map.imshow(mapa_cont, origin='lower', cmap='gray', vmin=0, vmax=1, aspect='equal', extent=map_extent)

    ax_map.set_xlim(zoom_x[0], zoom_x[1])
    ax_map.set_ylim(zoom_y[0], zoom_y[1])
    ax_map.axvline(x=x_cen, color='k', linestyle='--', linewidth=2)

    # Anotações de Posição
    offset_x_texto, offset_ponta = 17, 2
    props_seta = dict(facecolor='red', edgecolor='red', shrink=0.05, width=2, headwidth=8)

    ax_map.annotate('Posição 1', xy=(x_cen + offset_ponta, pos_fenda1), xytext=(x_cen + offset_x_texto, pos_fenda1), 
                    arrowprops=props_seta, color='red', fontsize=14, weight='bold', va='center', ha='left')
    ax_map.annotate('Posição 2', xy=(x_cen + offset_ponta, pos_fenda2), xytext=(x_cen + offset_x_texto, pos_fenda2), 
                    arrowprops=props_seta, color='red', fontsize=14, weight='bold', va='center', ha='left')

    ax_map.plot(x_cen, pos_fenda1, marker='x', color='red', markersize=12, markeredgewidth=3)
    ax_map.plot(x_cen, pos_fenda2, marker='x', color='red', markersize=12, markeredgewidth=3)

    # ================================================================
    # CÍRCULOS ATUALIZADOS PARA RECEBER OS PARÂMETROS
    # ================================================================
    regioes = [
        {'raio': raio_umbra,    'cor': 'g', 'label': 'Umbra'},
        {'raio': raio_penumbra, 'cor': 'b', 'label': 'Penumbra'}
    ]
    for r in regioes:
        circulo = patches.Circle((x_cen, y_cen), r['raio'], edgecolor=r['cor'],
                                 facecolor='none', linestyle='-', linewidth=2, label=r['label'])
        ax_map.add_patch(circulo)

    ax_map.set_xlabel('X [arcsec]', fontsize=18)
    ax_map.set_ylabel('Y [arcsec]', fontsize=18)
    ax_map.tick_params(axis='both', which='major', labelsize=14)

    cbar = fig.colorbar(im, cax=cax, ticks=np.arange(0.0, 1.1, 0.1), orientation='horizontal')
    cbar.set_label(r'$I/I_c$', fontsize=18, labelpad=10)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')

    formatter_us = ticker.FuncFormatter(lambda x, pos: f'{x:,.0f}')
    ax_map.xaxis.set_major_formatter(formatter_us)
    ax_map.yaxis.set_major_formatter(formatter_us)

    # PAINEL DA DIREITA: SUBPLOTS DE STOKES
    titulos_stokes = [r"Stokes $I/I_c$", r"Stokes $Q/I_c$", r"Stokes $U/I_c$", r"Stokes $V/I_c$"]
    limites_stokes = [(0.0, 1.0), (-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)]

    def px_do_centro(B_gauss):
        desloc_lambda = 4.6686e-13 * 2.5 * (6302.5**2) * B_gauss
        return desloc_lambda / dispersao

    pixel_do_6302 = 52
    centro_x_6302_5 = pixel_do_6302 + (0.5 / dispersao)
    altura_y_regua = pos_fenda1 + 2

    valores_ticks_kG = [-4, -2, 0, 2, 4]
    posicoes_x_ticks = [centro_x_6302_5 + px_do_centro(val * 1000) for val in valores_ticks_kG]

    valores_textos_kG = [0, 2, 4]
    posicoes_x_textos = [centro_x_6302_5 + px_do_centro(val * 1000) for val in valores_textos_kG]

    for i in range(4):
        ax = axs_stokes[i]
        
        ax.set_xticks([5, 52, 98])
        ax.set_xticklabels(['6301', '6302', '6303'])

        passo_esq = (52 - 5) / 10
        passo_dir = (98 - 52) / 10
        
        ticks_antes = [5 - j * passo_esq for j in range(1, 3) if (5 - j * passo_esq) >= 0]
        ticks_meio = [5 + j * passo_esq for j in range(1, 10)] + [52 + j * passo_dir for j in range(1, 10)]
        ticks_depois = [98 + j * passo_dir for j in range(1, 5) if (98 + j * passo_dir) <= stokes_norm.shape[3]]
        
        minor_ticks_calculados = ticks_antes + ticks_meio + ticks_depois
        ax.set_xticks(minor_ticks_calculados, minor=True)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', left=False, labelleft=False)

        vmin, vmax = limites_stokes[i]
        dados = stokes_norm[:,idx_x_fenda, i, :]
        im_st = ax.imshow(dados, origin='lower', cmap='gray', extent=extent_espectral, aspect='auto', vmin=vmin, vmax=vmax)

        ticks_cb = [0.0, 0.5, 1.0] if i == 0 else [-0.2, 0.0, 0.2]
        cb = fig.colorbar(im_st, cax=caxs_stokes[i], ticks=ticks_cb, orientation='horizontal')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.tick_params(labelsize=11)
        cb.ax.set_xticklabels([str(t) for t in ticks_cb])
        
        labels = cb.ax.get_xticklabels()
        if labels:
            labels[0].set_horizontalalignment('left')
            labels[-1].set_horizontalalignment('right')
        caxs_stokes[i].set_title(titulos_stokes[i], fontsize=18, pad=12)

        ax.axhline(y=pos_fenda1, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.axhline(y=pos_fenda2, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

        # Desenha a régua Zeeman
        ax.plot([posicoes_x_ticks[0], posicoes_x_ticks[-1]], [altura_y_regua, altura_y_regua], color='cyan', linewidth=1.5, alpha=0.8)
        for val_tick, px_x in zip(valores_ticks_kG, posicoes_x_ticks):
            tamanho_traco = 18 if val_tick == 0 else 10
            ax.plot([px_x], [altura_y_regua], color='cyan', marker='|', markersize=tamanho_traco)

    # Anotações finais exclusivas de I
    axs_stokes[0].text(8, pos_fenda1 + 0.8, '1', color='red', fontsize=14, weight='bold', va='bottom')
    axs_stokes[0].text(8, pos_fenda2 + 0.8, '2', color='red', fontsize=14, weight='bold', va='bottom')

    for val, px_x in zip(valores_textos_kG, posicoes_x_textos):
        axs_stokes[0].text(px_x, altura_y_regua + 2, str(val), color='cyan', fontsize=10, ha='center', va='bottom')

    axs_stokes[0].text(posicoes_x_textos[-1] + 2, altura_y_regua + 2, "(kG)", color='cyan', fontsize=10, ha='left', va='bottom')

    ax_map.text(1, 1.2, 'a) Intensidade do Contínuo', transform=ax_map.transAxes, fontsize=20, weight='bold', va='bottom', ha='right')
    fig.text(0.725, 0.94, 'b) Espectro de Stokes Completo', transform=fig.transFigure, fontsize=20, weight='bold', va='bottom', ha='center')
    fig.text(0.62, 0.1, r"Comprimento de onda ($\AA$)", fontsize=18, ha='center')

    plt.subplots_adjust(left=0.06, right=0.96, bottom=0.15, top=0.84, wspace=0.03, hspace=0.05)

    # Corrige alinhamento do cax
    fig.canvas.draw()
    pos_cax = cax.get_position()
    fator_largura = 1.69
    nova_largura = pos_cax.width * fator_largura
    novo_x0 = pos_cax.x0 + (pos_cax.width - nova_largura) / 2
    cax.set_position([novo_x0, pos_cax.y0, nova_largura, pos_cax.height])

    if save_fig:
        plt.savefig(filename, dpi=600, transparent=True, bbox_inches='tight', pad_inches=0.1)
        
    return fig
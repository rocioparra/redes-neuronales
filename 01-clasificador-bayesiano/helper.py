import pandas as pd
import numpy as np
import bisect
import scipy.stats as stats
from matplotlib import pyplot as plt


N_BINS = 30
BIN_SIZE = np.array([5, 0.1])

FIGSIZE = (10, 5)

errors = pd.DataFrame(columns=['Peso', 'Altura', 'Genero'])


def head(n=5):
    return df.head(n)
    

def get_bin(a, x):
    ind = bisect.bisect(a, x)
    if ind == len(a):
        ind -= 1
    return ind


class WeightHeightData:
    def __init__(self, name, df, dataedges=None):
        self.df = pd.DataFrame(df[['Peso', 'Altura', 'Genero']])
        self.name = str(name)
        
        ## histogram from samples
        bins = dataedges if dataedges is not None else N_BINS
        self.hist, self.pedges, self.aedges = np.histogram2d(
            self.df['Peso'], self.df['Altura'], bins=bins)
        
        ## pdf assuming joint gaussians
        self.cov = df[['Peso', 'Altura']].cov()
        self.mean = df[['Peso', 'Altura']].mean()
        
        self.dist = stats.multivariate_normal(mean=self.mean, cov=self.cov)
        
        return
    
    def __len__(self):
        return len(self.df)


df = pd.read_csv("https://raw.githubusercontent.com/lab-pep-itba/Clase-3---Clasificadores-Bayesianos/master/data/alturas-pesos-mils-train.csv", index_col="Unnamed: 0")
df_test = pd.read_csv("https://raw.githubusercontent.com/lab-pep-itba/Clase-3---Clasificadores-Bayesianos/master/data/alturas-pesos-mils-test.csv", index_col="Unnamed: 0")

GRUPOS = [WeightHeightData('Toda la población', df)]
GRUPOS.extend([
    WeightHeightData('Mujeres', df[df['Genero']=='Mujer'],
                    dataedges=(GRUPOS[0].pedges, GRUPOS[0].aedges)),
    WeightHeightData('Hombres', df[df['Genero']=='Hombre'],
                    dataedges=(GRUPOS[0].pedges, GRUPOS[0].aedges))
])


def get_prob_gaussian(data, x):
    return data.dist.cdf(x+BIN_SIZE/2)-data.dist.cdf(x-BIN_SIZE/2)
    

def get_prob_hist(data, x):
    return data.hist[get_bin(data.pedges, x[0])][get_bin(data.aedges, x[1])]
    

def plot_hist_and_gaussian():
    for grupo in GRUPOS:
        _plot_hist_and_gaussian(grupo)
    

def _plot_hist_and_gaussian(data):
    # create axis
    _, axes = plt.subplots(
        1, 2, figsize=FIGSIZE, gridspec_kw={'width_ratios': [1, 1]}, sharey='row')

    # plot histogram 
    axes[0].imshow(data.hist, origin='lower', extent=(
        data.pedges[0], data.pedges[-1], data.aedges[0], data.aedges[-1]))
    
    # plot gaussian pdf
    pesos, alturas = np.mgrid[data.pedges[0]:data.pedges[-1]:1,
                              data.aedges[0]:data.aedges[-1]:1]
    pa = np.dstack((pesos, alturas))
    axes[1].contourf(pesos, alturas, data.dist.pdf(pa))
    
    # final plot settings
    for i, title in enumerate(['Histograma a partir de muestras', 
                            'Curvas de nivel de pdf gaussiana']):
        axes[i].set_title(title)
        axes[i].set_xlabel('Peso [kg]')
        axes[i].axis('scaled')
    axes[0].set_ylabel('Altura [cm]')
    
    # show results
    print(f'\nGráficos correspondientes a {data.name.lower()}')
    plt.show()


def get_stats():
    cols = ['Grupo', 'Probabilidad', 'Media peso', 'Desvío peso', 
            'Media altura', 'Desvío altura', 'Correlación peso-altura']
    
    stats = pd.DataFrame(columns=cols)
    
    n = len(GRUPOS[0])
    for grupo in GRUPOS:
        p = len(grupo)/n
        std_p = np.sqrt(grupo.cov['Peso']['Peso'])
        std_a = np.sqrt(grupo.cov['Altura']['Altura'])
        rho = grupo.cov['Peso']['Altura']/std_p/std_a
        
        row = [grupo.name, p, grupo.mean['Peso'], std_p, grupo.mean['Altura'], std_a, rho]
        row = pd.Series(row, index=cols)
        stats = stats.append(row, ignore_index=True, sort=False)
    
    return stats

    
def test(method='gaussian'):
    todos, mujeres, hombres = GRUPOS

    global errors

    p_hombre = len(hombres)/len(todos)
    get_prob = get_prob_gaussian if method == 'gaussian' else get_prob_hist
    
    errors = pd.DataFrame(columns=['Peso', 'Altura', 'Genero'])
    aciertos = 0
    for _, persona in df_test.iterrows():
        x = (persona['Peso'], persona['Altura'])
        p_pa_h = get_prob(hombres, x)
        p_pa_m = get_prob(mujeres, x)
        p_pa = p_pa_h + p_pa_m
        
        if p_pa != 0:
            p_h = p_pa_h * p_hombre / p_pa
            p_m = p_pa_m * (1-p_hombre) / p_pa
        else:
            p_h, p_m = (1, 0) if persona['Altura'] >= todos.mean['Altura'] else (0, 1)

        guess = 'Hombre' if p_h >= p_m else 'Mujer'
        if persona['Genero'] == guess:
            aciertos += 1
        else:
            errors = errors.append(persona)
        
    return aciertos/len(df_test), len(errors), len(errors[errors.Genero == 'Mujer'])/len(errors)

    
def plot_errors():
    errors_m = errors[errors.Genero == 'Mujer']
    errors_h = errors[errors.Genero == 'Hombre']
    e = [errors_m, errors_h]
    bns = (GRUPOS[0].pedges, GRUPOS[0].aedges)
    _, axes = plt.subplots(
        1, 2, figsize=FIGSIZE, gridspec_kw={'width_ratios': [1, 1]}, sharey='row')

    # final plot settings
    for i, title in enumerate(['Mujeres clasificadas como hombres', 
                               'Hombres clasificados como mujeres']):

        hist, pedges, aedges = np.histogram2d(e[i]['Peso'], e[i]['Altura'], bins=bns)
        axes[i].imshow(hist, origin='lower', extent=(
            pedges[0], pedges[-1], aedges[0], aedges[-1])
        )
        axes[i].set_title(title)
        axes[i].set_xlabel('Peso [kg]')
        axes[i].axis('scaled')
    axes[0].set_ylabel('Altura [cm]')
    
    # show results
    plt.show()
    

if __name__ == '__main__':
    right_p, wrong_n, wrong_w_p = test()
    print(right_p)
    plot_errors()

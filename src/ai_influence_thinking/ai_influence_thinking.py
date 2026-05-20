import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# =========================
# COLOR PALETTE
# =========================

NAVY      = "#333399"
RED       = "#C0504D"
GREEN     = "#9BBB59"
PURPLE    = "#8064A2"
TEAL      = "#4BACC6"
ORANGE    = "#F79646"
BLUE_MID  = "#4F81BD"

WHITE     = "#FFFFFF"
DARK_TEXT = "#1F497D"

BG        = "#F2F2F2"
PLOT_BG   = "#EBEBEB"
GRID      = "#D9D9D9"

PALETTE_5 = [NAVY, RED, GREEN, PURPLE, TEAL]

# =========================
# STYLE
# =========================

plt.style.use('default')

CLEAN_PATH = '../../data/clean/ai_students_clean.csv'
df = pd.read_csv(CLEAN_PATH)

queue = ['Never', 'Rarely', 'Sometimes', 'Frequently', 'Always']

df['uses_ai_for_assignments'] = pd.Categorical(
    df['uses_ai_for_assignments'],
    queue,
    ordered=True
)

queue_colors = PALETTE_5


def setup_axis(ax, title, xlabel, ylabel):

    ax.set_title(
        title,
        fontsize=14,
        fontweight='bold',
        pad=18,
        color=DARK_TEXT
    )

    ax.set_xlabel(
        xlabel,
        fontsize=12,
        color=DARK_TEXT,
        labelpad=10
    )

    ax.set_ylabel(
        ylabel,
        fontsize=12,
        color=DARK_TEXT,
        labelpad=10
    )

    ax.tick_params(
        colors=DARK_TEXT,
        labelsize=11
    )

    ax.set_facecolor(PLOT_BG)


fig, (ax_heat, ax_ridge) = plt.subplots(
    1, 2,
    figsize=(22, 9),
    gridspec_kw={
        'width_ratios': [1.1, 1],
        'wspace': 0.09
    }
)

fig.patch.set_facecolor(BG)

# =========================
# HEATMAP
# =========================

crossT = (
    pd.crosstab(
        df['uses_ai_for_assignments'],
        df['ai_replaces_own_thinking_score'],
        normalize='index'
    ) * 100
).rename(columns=lambda c: f'Score {c}')

sns.heatmap(
    crossT,
    annot=True,
    fmt='.1f',
    cmap='RdPu',
    linewidths=2,
    linecolor=BG,
    annot_kws={
        'size': 13,
        'weight': 'bold'
    },
    cbar_kws={
        'label': '% studentesh',
        'shrink': 0.6
    },
    ax=ax_heat
)

setup_axis(
    ax_heat,
    'Si ndryshon te menduarit\nsipas sasise se perdorimit?',
    'Zevendesimi i mendimit (1=aspak → 5=plotesisht)',
    'Perdorimi i AI per detyra'
)

# =========================
# BUBBLE CHART
# =========================

mesataret = (
    df.groupby('uses_ai_for_assignments', observed=False)
      .agg(
          ore=('daily_ai_tool_usage_hrs', 'mean'),
          varesia=('ai_dependency_score', 'mean'),
          zevendesimi=('ai_replaces_own_thinking_score', 'mean')
      )
      .reset_index()
)

for ore, varesia, zev, label, color in zip(
    mesataret['ore'],
    mesataret['varesia'],
    mesataret['zevendesimi'],
    mesataret['uses_ai_for_assignments'],
    queue_colors
):

    ax_ridge.scatter(
        ore,
        varesia,
        s=zev * 600,
        color=color,
        alpha=0.9,
        zorder=4,
        edgecolors=WHITE,
        linewidth=1.5
    )

    ax_ridge.text(
        ore,
        varesia + 0.48,
        label,
        ha='center',
        va='bottom',
        fontsize=11,
        fontweight='bold',
        color=DARK_TEXT,
        zorder=5
    )

    ax_ridge.text(
        ore,
        varesia,
        f'{zev:.1f}',
        ha='center',
        va='center',
        fontsize=10,
        fontweight='bold',
        color=WHITE,
        zorder=5
    )

ax_ridge.plot(
    mesataret['ore'],
    mesataret['varesia'],
    color=BLUE_MID,
    linewidth=1.5,
    linestyle='--',
    zorder=3
)

ax_ridge.set(
    xlim=(-0.3, 7.2),
    ylim=(1.8, 9.5)
)

ax_ridge.xaxis.grid(True, color=GRID)
ax_ridge.yaxis.grid(True, color=GRID)

setup_axis(
    ax_ridge,
    'Sa me shume AI, aq me pak pavaresi ne te menduar',
    'Mesatarja e oreve ditore te perdorimit te AI',
    'Mesatarja e varesise nga AI (1-10)'
)

# =========================
# TITULLI KRYESOR
# =========================ë

fig.suptitle(
    'Ndikimi i AI ne pavaresine e te menduarit te studenteve',
    fontsize=19,
    fontweight='bold',
    y=1.02,
    color=NAVY
)

plt.savefig(
    'ai_influence_chart.png',
    dpi=160,
    bbox_inches='tight',
    facecolor=BG
)

plt.close()

print('Saved: ai_influence_chart.png')
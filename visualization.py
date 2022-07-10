import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import tqdm

from model.utils import *
from envrioment import DHP_HLR, GRU_HLR

plt.style.use('seaborn-whitegrid')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.5, y=1.5, z=1.25)
)


def raw_data_visualize():
    raw = pd.read_csv('./data/opensource_dataset_p_history.tsv', sep='\t')
    raw.dropna(inplace=True)
    raw = raw[raw['group_cnt'] > 1000]
    raw['label'] = raw['r_history'] + '/' + raw['t_history']

    fig = px.scatter_3d(raw, x='last_p_recall', y='last_halflife',
                        z='halflife', color='d',
                        hover_name='label')
    fig.layout.scene.xaxis.title = r"p<sub>i-1</sub>"
    fig.layout.scene.yaxis.title = r"h<sub>i-1</sub>"
    fig.layout.scene.zaxis.title = r"h<sub>i</sub>"
    h_array = np.arange(0.5, 1600, 5)  # 03
    p_array = np.arange(0.3, 0.97, 0.05)  # 03
    h_array, p_array = np.meshgrid(h_array, p_array)
    fig.add_surface(y=h_array, x=p_array, z=h_array, showscale=False)
    fig.update_traces(opacity=0.2, selector=dict(type='surface'))
    fig.layout.scene.yaxis.type = 'log'
    fig.layout.scene.zaxis.type = 'log'
    fig.update_traces(marker_size=2, selector=dict(type='scatter3d'))
    fig.update_scenes(xaxis_autorange="reversed")
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_coloraxes(colorbar_tickfont_size=24)
    fig.write_image(f"plot/DHP_model_raw.pdf", width=1000, height=1000)
    fig.show()


def dhp_model_visualize():
    model = DHP_HLR()
    h_array = np.arange(0.5, 750.5, 1)  # 03
    p_array = np.arange(0.97, 0.3, -0.01)  # 03
    h_array, p_array = np.meshgrid(h_array, p_array)
    surface = [
        go.Surface(x=h_array, y=p_array, z=model.cal_recall_halflife(diff, h_array, p_array),
                   showscale=True if diff == 10 else False
                   , cmin=0, cmax=6500
                   )
        for diff in
        range(1, 11)]
    fig = go.Figure(data=surface)
    fig.layout.scene.xaxis.title = r"h<sub>i-1</sub>"
    fig.layout.scene.yaxis.title = r"p<sub>i-1</sub>"
    fig.layout.scene.zaxis.title = r"h<sub>i</sub>"
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_traces(colorbar_title_text='', colorbar_tickfont_size=24, selector=dict(type='surface'))
    # fig.write_html(f"./plot/DHP_recall_model.html")
    fig.write_image(f"./plot/DHP_recall_model.pdf", width=1000, height=1000)
    # fig.show()
    surface = [
        go.Surface(x=h_array, y=p_array, z=model.cal_recall_halflife(diff, h_array, p_array) / h_array
                   , showscale=True if diff == 10 else False
                   , cmin=0, cmax=25
                   ) for diff in
        range(1, 11)]
    fig = go.Figure(data=surface)
    fig.layout.scene.xaxis.title = r"h<sub>i-1</sub>"
    fig.layout.scene.yaxis.title = r"p<sub>i-1</sub>"
    fig.layout.scene.zaxis.title = r"h<sub>i</sub>/h<sub>i-1</sub>"
    # fig.write_html(f"./plot/DHP_recall_inc_model.html")
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_traces(colorbar_title_text='', colorbar_tickfont_size=24, selector=dict(type='surface'))
    fig.write_image(f"./plot/DHP_recall_inc_model.pdf", width=1000, height=1000)
    fig.layout.scene.zaxis.type = 'log'
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=3), ))
    fig.update_traces(
        cmin=0,
        cmax=1.45,
        colorbar=dict(
            tickvals=[i for i in np.arange(0, 1.45, 0.2)],
            ticktext=[round(np.power(10, i)) for i in np.arange(0, 1.45, 0.2)]
        )
    )
    fig.write_image(f"./plot/DHP_recall_inc_log_model.pdf", width=1000, height=1000)
    # fig.show()
    surface = [
        go.Surface(x=h_array, y=p_array, z=model.cal_forget_halflife(diff, h_array, p_array)
                   , showscale=True if diff == 10 else False
                   , cmin=0, cmax=16
                   ) for diff in
        range(1, 11)]
    fig = go.Figure(data=surface)
    fig.layout.scene.xaxis.title = r"h<sub>i-1</sub>"
    fig.layout.scene.yaxis.title = r"p<sub>i-1</sub>"
    fig.layout.scene.zaxis.title = r"h<sub>i</sub>"
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_traces(colorbar_title_text='', colorbar_tickfont_size=24, selector=dict(type='surface'))
    # fig.write_html(f"./plot/DHP_forget_model.html")
    fig.write_image(f"./plot/DHP_forget_model.pdf", width=1000, height=1000)
    # fig.show()


def dhp_policy_action_visualize():
    delta_t = np.load('./SSP-MMC/dhp_policy.npy')[:, 40:-2]
    halflife = np.array([np.power(1.05, i) for i in range(121)])
    d = np.arange(1, 19, 1)
    halflife, d = np.meshgrid(halflife, d)
    fig = go.Figure(data=go.Surface(x=d, y=halflife, z=delta_t))
    fig.update_scenes(xaxis_autorange="reversed")
    fig.update_scenes(yaxis_autorange="reversed")
    fig.update_layout(scene=dict(
        xaxis_title='d',
        yaxis_title='halflife',
        zaxis_title='delta_t'))
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    # fig.update_traces(colorbar_tickfont_size=18, selector=dict(type='surface'))
    fig.write_image(f"./plot/dhp_policy_delta_t.pdf", width=1000, height=1000)

    cost = np.load('./SSP-MMC/dhp_cost.npy')[:, 40:-2]
    fig = go.Figure(data=go.Surface(x=d, y=halflife, z=cost))
    fig.update_scenes(xaxis_autorange="reversed")
    fig.update_layout(scene=dict(
        xaxis_title='d',
        yaxis_title='halflife',
        zaxis_title='cost'))
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_traces(colorbar_tickfont_size=18, selector=dict(type='surface'))
    fig.write_image(f"./plot/dhp_policy_cost.pdf", width=1000, height=1000)
    # fig.show()

    p_recall = np.load('./SSP-MMC/dhp_recall.npy')[:, 40:-2]
    fig = go.Figure(data=go.Surface(x=d, y=halflife, z=p_recall, surfacecolor=-np.log(1 - p_recall)))
    fig.update_traces(
        colorbar=dict(
            tickvals=[i for i in np.arange(0, 6, 0.5)],
            ticktext=[-round(np.exp(-i) - 1, 3) for i in np.arange(0, 6, 0.5)]
        )
    )
    fig.update_scenes(xaxis_autorange="reversed")
    fig.update_scenes(yaxis_autorange="reversed")
    fig.update_layout(scene=dict(
        xaxis_title='d',
        yaxis_title='halflife',
        zaxis_title='p_recall'))
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_traces(colorbar_tickfont_size=18, selector=dict(type='surface'))
    fig.write_image(f"./plot/dhp_policy_p_recall.pdf", width=1000, height=1000)


def gru_policy_action_visualize():
    halflife = np.load('./SSP-MMC/gru_half_life.npy')
    s1, s2 = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))
    fig = go.Figure(data=go.Surface(x=s1, y=s2, z=halflife, surfacecolor=np.log(halflife)))
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))
    fig.update_traces(line_width=5, selector=dict(type='contour'))
    fig.layout.scene.zaxis.type = 'log'
    fig.update_layout(scene=dict(
        xaxis_title='s1',
        yaxis_title='s2',
        zaxis_title='halflife'))
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_traces(colorbar_tickfont_size=18, selector=dict(type='surface'))
    fig.update_traces(
        colorbar=dict(
            tickvals=[i for i in range(-2, 7)],
            ticktext=[round(np.exp(i), 1) for i in range(-2, 7)]
        )
    )
    fig.write_image(f"./plot/gru_policy_halflife.pdf", width=1000, height=1000)
    cost = np.load('./SSP-MMC/gru_cost.npy')
    fig = go.Figure(data=go.Surface(x=s1, y=s2, z=cost))
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))
    fig.update_layout(scene=dict(
        xaxis_title='s1',
        yaxis_title='s2',
        zaxis_title='cost'))
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_traces(colorbar_tickfont_size=18, selector=dict(type='surface'))
    fig.write_image(f"./plot/gru_policy_cost.pdf", width=1000, height=1000)
    recall = np.load('./SSP-MMC/gru_recall.npy')
    df = pd.DataFrame(recall) \
        .replace(0, np.nan) \
        .ffill().fillna(0.99)
    recall = df.values
    fig = go.Figure(data=go.Surface(x=s1, y=s2, z=recall, surfacecolor=-np.log(1 - recall)))
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))

    fig.update_traces(
        colorbar=dict(
            tickvals=[i for i in np.arange(0, 6, 0.5)],
            ticktext=[-round(np.exp(-i) - 1, 3) for i in np.arange(0, 6, 0.5)]
        )
    )
    fig.update_layout(scene=dict(
        xaxis_title='s1',
        yaxis_title='s2',
        zaxis_title='p_recall'))
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_traces(colorbar_tickfont_size=18, selector=dict(type='surface'))
    fig.update_scenes(aspectmode='cube')
    fig.write_image(f"./plot/gru_policy_p_recall.pdf", width=1000, height=1000)
    policy = np.load('./SSP-MMC/gru_policy.npy')
    policy[np.where(policy == 0)] = 1
    fig = go.Figure(data=go.Surface(x=s1, y=s2, z=policy, surfacecolor=np.log(policy)))
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))
    fig.layout.scene.zaxis.type = 'log'
    fig.update_layout(scene=dict(
        xaxis_title='s1',
        yaxis_title='s2',
        zaxis_title='delta_t'))
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_traces(colorbar_tickfont_size=18, selector=dict(type='surface'))
    fig.update_traces(
        colorbar=dict(
            tickvals=[i for i in range(0, 6)],
            ticktext=[round(np.exp(i), 1) for i in range(0, 6)]
        )
    )
    fig.write_image(f"./plot/gru_policy_delta_t.pdf", width=1000, height=1000)
    # fig.show()


def gru_model_visualize():
    model = GRU_HLR()
    # for name, param in model.named_parameters():
    #     print(name, param)
    recall_record = np.array([0, 0, 0])
    forget_record = np.array([0, 0, 0])
    for s1, s2 in tqdm.tqdm([[x0, y0] for x0 in np.arange(-1, 1, 0.02) for y0 in np.arange(-1, 1, 0.02)]):
        h = model.state2halflife(np.array([s1, s2]))
        # # print(f'current state: {s1:.2f} {s2:.2f}\thalflife: {h: .2f}')
        # for t in np.arange(1, max(1, round(2 * h)) + 1, max(1, round(h / 10))):
        #     p = np.exp2(-t / h)
        #     n_state, nh = model.next_state(np.array([s1, s2]), 1, t, p)
        #     ns1, ns2 = n_state
        for p in np.arange(0.35, 0.96, 0.05):
            t = int(np.round(- np.log2(p) * h))
            p = np.exp2(-t / h)
            if t < 1 or p < 0.35:
                continue
            n_state, nh = model.next_state(np.array([s1, s2]), 1, t, p)
            # print(f'delta_t: {t}\tp_recall: {p: .3f}\tnext state: {ns1:.2f} {ns2:.2f}\thalflife: {nh: .2f}')
            recall_record = np.vstack((recall_record, np.array([h, p, nh])))
            n_state, nh = model.next_state(np.array([s1, s2]), 0, t, p)
            forget_record = np.vstack((forget_record, np.array([h, p, nh])))
    # print(record[1:, :])
    recall_model = pd.DataFrame(data=recall_record[1:, :], columns=['last_halflife', 'last_p_recall', 'halflife'])
    recall_model.drop_duplicates(inplace=True)
    recall_model['halflife_increase'] = recall_model['halflife'] / recall_model['last_halflife']
    recall_model['halflife_increase_log'] = recall_model['halflife_increase'].map(np.log)
    # recall_model['last_p_recall'] = recall_model['last_p_recall'].map(lambda x: np.round(x, decimals=2))
    # recall_model['last_halflife'] = recall_model['last_halflife'].map(lambda x: np.round(x, decimals=2))
    # last_halflife = recall_model['last_halflife'].drop_duplicates().values
    # last_p_recall = recall_model['last_p_recall'].drop_duplicates().values
    # last_halflife, last_p_recall = np.meshgrid(last_halflife, last_p_recall)
    # halflife = np.empty(last_halflife.shape)
    # halflife[:] = np.nan
    # for i in range(last_halflife.shape[0]):
    #     for j in range(last_halflife.shape[1]):
    #         halflife[i, j] = recall_model[(recall_model['last_halflife'] == last_halflife[i, j]) & (
    #                 recall_model['last_p_recall'] == last_p_recall[i, j])]['halflife'].mean()
    #
    # fig = go.Figure(data=[go.Surface(z=halflife, x=last_halflife, y=last_p_recall)])
    # fig.show()
    # exit()

    fig = px.scatter_3d(recall_model, x='last_halflife', y='last_p_recall', z='halflife', color='halflife')
    fig.layout.scene.xaxis.title = r"h<sub>i-1</sub>"
    fig.layout.scene.yaxis.title = r"p<sub>i-1</sub>"
    fig.layout.scene.zaxis.title = r"h<sub>i</sub>"
    fig.update_traces(marker_size=2, selector=dict(type='scatter3d'))
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_coloraxes(colorbar_tickfont_size=24)
    fig.update_coloraxes(colorbar_title_text='')
    fig.write_image('./plot/GRU_recall_model.pdf', width=1000, height=1000)
    # fig.show()

    fig = px.scatter_3d(recall_model, x='last_halflife', y='last_p_recall', z='halflife_increase',
                        color='halflife_increase_log')
    fig.layout.scene.xaxis.title = r"h<sub>i-1</sub>"
    fig.layout.scene.yaxis.title = r"p<sub>i-1</sub>"
    fig.layout.scene.zaxis.title = r"h<sub>i</sub>/h<sub>i-1</sub>"
    fig.update_traces(marker_size=2, selector=dict(type='scatter3d'))
    fig.update_coloraxes(colorbar_tickmode='array', colorbar_tickvals=[i for i in np.arange(0, 3.5, 0.5)],
                         colorbar_ticktext=[round(np.exp(i), 1) for i in np.arange(0, 3.5, 0.5)])
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_coloraxes(colorbar_tickfont_size=24)
    fig.update_coloraxes(colorbar_title_text='')
    fig.write_image('./plot/GRU_recall_inc_model.pdf', width=1000, height=1000)
    # fig.show()
    fig.layout.scene.zaxis.type = 'log'
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=3), ))
    fig.write_image('./plot/GRU_recall_inc_log_model.pdf', width=1000, height=1000)
    # fig.show()

    forget_model = pd.DataFrame(data=forget_record[1:, :], columns=['last_halflife', 'last_p_recall', 'halflife'])
    forget_model.drop_duplicates(inplace=True)

    fig = px.scatter_3d(forget_model, x='last_halflife', y='last_p_recall', z='halflife', color='halflife')
    fig.layout.scene.xaxis.title = r"h<sub>i-1</sub>"
    fig.layout.scene.yaxis.title = r"p<sub>i-1</sub>"
    fig.layout.scene.zaxis.title = r"h<sub>i</sub>"
    fig.update_traces(marker_size=2, selector=dict(type='scatter3d'))
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_coloraxes(colorbar_tickfont_size=24)
    fig.update_coloraxes(colorbar_title_text='')
    fig.write_image('./plot/GRU_forget_model.pdf', width=1000, height=1000)
    # fig.show()


if __name__ == "__main__":
    # raw_data_visualize()
    # dhp_model_visualize()
    gru_model_visualize()
    # dhp_policy_action_visualize()
    # gru_policy_action_visualize()

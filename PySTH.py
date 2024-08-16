import os
import xlwt
from tabulate import tabulate
import pandas as pd
from matplotlib import ticker, cm
import xlrd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FixedFormatter
import matplotlib as mpl
import numpy as np
# import matplotlib.font_manager as fm
# fm.fontManager.addfont('times(1).ttf')

def coSTH(VLC,xh,xo,s,E):
    if not os.path.exists(s):
        os.makedirs(s)
    # print(len(E))
    len1 = int(len(E))
    workbook1 = xlrd.open_workbook(r'D:\down\1.xls')  # 文件路径读文件
    names1 = workbook1.sheet_names()
    # print(names1)
    objsth = workbook1.sheet_by_name("Sheet1")
    if VLC>=1.72:
        Xh = np.linspace(0, 2, 201)
        Xo = np.linspace(0, 2, 201)
    elif VLC<1.72:
        # Xh = np.linspace(0, 1, 101)
        # Xo = np.linspace(0, 1, 101)
        Xh = np.linspace(0, 2, 201)
        Xo = np.linspace(0, 2, 201)

    # VLC = 0
    t0 = E[0]
    xh0 = xh - (t0 * 0.059)
    xo0 = xo + (t0 * 0.059)
    xh = round(xh,2)
    xo = round(xo,2)
    t1 = E[-1]

    xh1 = xh - (t1 * 0.059)
    xo1 = xo + (t1 * 0.059)

    xh1 = round(xh1, 2)
    xo1 = round(xo1, 2)


    X , Y = np.meshgrid(Xh, Xo)

    Eg = X + Y + 1.23 - VLC

    l = objsth.col_values(2)
    f = objsth.col_values(3)
    h = objsth.col_values(4)
    n = objsth.col_values(5)
    # 第2行开始
    l.pop(0)
    f.pop(0)
    h.pop(0)
    n.pop(0)
    Emax = np.max(l)
    emax = np.max(h)
    # 求最小梯度
    DEmin = np.min(np.gradient(l))
    dEmin = np.min(np.gradient(h))
    r = []
    Z = []
    max = 0
    min = 1000
    ans = 0
    ans1 = 0
    xhh = 0
    xoo = 0
    for i, Eg0 in zip(Xo, Eg):
        # ans = 0
        for j, Eg1 in zip(Xh, Eg0):
            # print(i,j,Eg1)
            if Eg1 < 0:
                r.append(np.nan)
                continue
            if j >= 0.2 and i >= 0.6:
                if Eg1 < 0.31:
                    Eg1 = 0.31
                E = Eg1
            elif j < 0.2 and i >= 0.6:
                E = Eg1 + 0.2 - j
                if E < 0.31:
                    E = 0.31
                if Eg1 < 0.31:
                    Eg1 = 0.31
            elif j >= 0.2 and i < 0.6:
                E = Eg1 + 0.6 - i
                if E < 0.31:
                    E = 0.31
                if Eg1 < 0.31:
                    Eg1 = 0.31
            elif j < 0.2 and i < 0.6:
                E = Eg1 + 0.8 - i - j
                if E < 0.31:
                    E = 0.31
                if Eg1 < 0.31:
                    Eg1 = 0.31
            # 插值积分

            Eintp = np.arange(Eg1, DEmin + Emax, DEmin)  # 无除
            eintp = np.arange(E, emax + dEmin, dEmin)  # 有除
            Jintp = np.interp(Eintp, l, f)
            jintp = np.interp(eintp, h, n)
            fintp = np.interp(Eintp, h, n)
            Egg = np.trapz(Jintp, Eintp)
            Eh = np.trapz(jintp, eintp)
            Ef = np.trapz(fintp, Eintp)

            nabs = Egg / 1000.37
            ncu = Eh * 1.23 / Egg
            STH = nabs * ncu
            correctedSTH = STH * 1000.37 / (1000.37 + VLC * Ef)
            if xh == round(j,2) and xo == round(i,2):

                ans = correctedSTH * 100
            if xh1 == round(j,2) and xo1 == round(i,2):

                ans1 = correctedSTH * 100
            if correctedSTH >= max:
                max = correctedSTH
                xhh = round(j,2)
                xoo = round(i,2)
            if correctedSTH < min:
                min = correctedSTH
            r.append(correctedSTH * 100)
        Z.append(r)
        r = []

    max = max * 100
    min = min * 100
    fig, ax = plt.subplots()
    # plt.figure(figsize=(12, 8))
    fig.set_size_inches(10, 8)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    colors = ["#86190d", "#dc0105", "#ffad0d", "#edfc1b", "#8efd72", "#1ffee1", "#03bffe", "#0c08ed"]
    # colors = ["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#da0300","#7e0004"]
    colors.reverse()
    cmap = LinearSegmentedColormap.from_list("custom_rainbow", colors)
    cmap_reversed = cmap.reversed()
    norm = mpl.colors.Normalize(vmin=min, vmax=max)
    # mpl.rc('font', size=33,  weight='bold')
    mpl.rc('font', size=33, family='Times New Roman', weight='bold')
    # ax.rc('font', size=12, family='serif')
    # mpl.rc('font', size=33, family='Times New Roman', weight='bold')
    # 刻度线值, manual=manual_locations
    # manual_locations = [(0.23, 0.64), (0.3, 0.72), (0.42, 0.8), (0.5, 0.83), (0.8, 0.85)]
    C = ax.contour(X, Y, Z, colors='black', linewidths=0)

    cs = ax.contourf(X, Y, Z, 5, cmap=cmap)  # 画出等高图

    x_val, y_val = np.where(np.isclose(Z, max))
    # for i,j in zip(y_val,x_val):
    #     print(i,end=' ')
    #     print(j)

    # 在图中标记Z值为0.5的点
    # plt.scatter(xh, xo, color='blue')  # 用红色标出你要标记的点
    # title = ["pH", '\u03c7h', '\u03c7o', "\u03B7abs", "\u03B7cu", "\u03B7STH", '\u03B7\u2032STH']

    if VLC == 0:
        # plt.annotate(f'{ans:.2f}%', (xh0, xo0), textcoords="offset points", xytext=(65, 10),
        #              ha='center', fontsize=15)
        # plt.annotate(f'\u03B7STH = {ans1:.2f}%', (xh1, xo1), textcoords="offset points", xytext=(30, 30),
        #              ha='center', fontsize=15)
        plt.annotate(f'$\eta_{{STH}}={ans:.2f}\%$', (xh0, xo0), textcoords="offset points", xytext=(10, 15),
                     ha='center', fontsize=25)
        ax.scatter(xh, xo, marker='*',c = 'black',s=30)
        plt.annotate(f'pH={0}', (xh0, xo0), textcoords="offset points", xytext=(0, -30),
                         ha='center', fontsize=25)
        plt.annotate(f'$\eta^{{max}}_{{STH}}={max:.2f}\%$', (0.3, 0.1),
                     textcoords="offset points", xytext=(80, -10), ha='center', fontsize=25, c='black', weight='bold')
    else:
        # plt.annotate(f'{ans:.2f}%', (xh0, xo0), textcoords="offset points", xytext=(0, -40),
        #              ha='center', fontsize=15)
        # plt.annotate(f'{ans1:.2f}%', (xh1, xo1), textcoords="offset points", xytext=(30, 30),
        #              ha='center', fontsize=15)
        # plt.annotate(f'$\eta^\prime_{{STH}}= {ans:.2f}\%$', (xh0, xo0), textcoords="offset points", xytext=(3, 20),
        #              ha='center', fontsize=30)
        ax.scatter(xh, xo, marker='*', c='black',s = 300)
        # plt.annotate(f'pH={0}', (xh0, xo0), textcoords="offset points", xytext=(0, -30),
        #              ha='center', fontsize=30)
        # plt.annotate(f'$\eta^{{max}}_{{STH}}={max:.2f}\%$', (X[x_val[-1], y_val[-1]], Y[x_val[-1], y_val[-1]]),
        #              textcoords="offset points", xytext=(80, 30), ha='center', fontsize=25, c='black', weight='bold')


    # if VLC == 0:
    #     plt.annotate(f'$\eta^{{max}}_{{STH}}={max:.2f}\%$', (0.3, 0.1),
    #                  textcoords="offset points", xytext=(40, -10), ha='center', fontsize=20, c='black', weight='bold')
    # else:
    #     plt.annotate(f'$\eta^{{max}}_{{STH}}={max:.2f}\%$', (X[x_val[-1], y_val[-1]], Y[x_val[-1], y_val[-1]]),
    #                  textcoords="offset points", xytext=(60, 30), ha='center', fontsize=20, c='black', weight='bold')


    if VLC == 0:
        for i in range(len(x_val)):
            ax.scatter(X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]], marker='\u002E', c='black',s = 2)
    else:
        for i in range(len(x_val)):
            ax.scatter(X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]], marker='*', c='white',s = 300)

    # ax.plot([X[x_val[0], y_val[0]] , X[x_val[-1],y_val[-1]], Y[x_val[-1], y_val[-1]],Y[x_val[-1], y_val[-1]]]
    #         , color='black', linestyle='-')
    # ax.plot([0, 0.2], [0.6, 0.6], color='black', linestyle='-')
    # ax.plot([0.2, 0.2], [0, 0.6], color='black', linestyle='-')
    # ax.plot([xh1, xh], [xo1, xo], color='black', linestyle='-')
    # ax.plot([xh1, xh], [xo1, xo], color='black', linestyle='-')
    # ax.plot([xh1, xh], [xo1, xo], color='black', linestyle='-')

    # if xh<2 and xo <2:
    #     for i in range(len1):
    #         xh_p = xh - (0.059 * i)
    #         xo_p = xo + (0.059 * i)
    #         xh_p = round(xh_p, 2)
    #         xo_p = round(xo_p, 2)
    #         if i == t0:
    #             plt.annotate(f'{i}', (xh_p, xo_p), textcoords="offset points", xytext=(30, 10),
    #                  ha='center', fontsize=15)
    #             plt.annotate(f'pH', (xh_p, xo_p), textcoords="offset points", xytext=(30,-10),
    #                          ha='center', fontsize=15)
    #         else:
    #             plt.annotate(f'{i}', (xh_p, xo_p), textcoords="offset points", xytext=(30, 10),
    #                          ha='center', fontsize=15)
    #         ax.scatter(xh_p, xo_p, marker='*',c = 'black')
    # if xh1 < 2 and xo1 < 2:
    #     plt.annotate(f'pH={t1}', (xh1, xo1), textcoords="offset points", xytext=(30, 0),
    #                  ha='center', fontsize=15)
    #     ax.scatter(xh1, xo1, marker='*', c='black')

    # ax.plot([xh1, xh0], [xo1, xo0], color='black', linestyle='-')


    # #添加colorbar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)  # 对colorbar的大小进行设置

    cbar.ax.invert_yaxis()

    cbar.update_ticks()  # 显示colorbar的刻度值


    font3 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 33,
             }
    ax.set_xlabel('\u03c7(H\u2082) (eV)', font3)
    ax.set_ylabel('\u03c7(O\u2082) (eV)', font3)
    # 设置主刻度的标签， 带入主刻度旋转角度和字体大小参数
    if VLC>=1.72:
        ax.set_xticks([0, 0.4,  0.8, 1.2,1.6,2])
        ax.set_xticklabels(['0', '0.4', '0.8','1.2', '1.6', '2'], fontsize=30, weight='bold', family='Times New Roman')
        ax.set_yticks([0,  0.4,  0.8,1.2,1.6,2])
        ax.set_yticklabels(['', '0.4', '0.8','1.2','1.6', '2'], fontsize=30, weight='bold', family='Times New Roman')

        cbar.ax.tick_params(width=3, length=8)
        ax.set_xticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
        ax.set_yticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
        ax.tick_params(direction='in', which='both')
        ax.tick_params(axis='x', which='major', width=3, length=8)
        ax.tick_params(axis='x', which='minor', width=2, length=4)
        ax.tick_params(axis='y', which='major', width=3, length=8)
        ax.tick_params(axis='y', which='minor', width=2, length=4)
    elif VLC<1.72:
        # ax.set_xticks([0, 0.2,  0.4, 0.6,0.8,1])
        # ax.set_xticklabels(['0', '0.2', '0.4','0.6', '0.8', '1'], fontsize=30, weight='bold', family='Times New Roman')
        # ax.set_yticks([0,  0.2,  0.4,0.6,0.8,1])
        # ax.set_yticklabels(['', '0.2', '0.4','0.6','0.8', '1'], fontsize=30, weight='bold', family='Times New Roman')
        #
        # cbar.ax.tick_params(width=3, length=8)
        # ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
        # ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
        # ax.tick_params(direction='in', which='both')
        # ax.tick_params(axis='x', which='major', width=3, length=8)
        # ax.tick_params(axis='x', which='minor', width=2, length=4)
        # ax.tick_params(axis='y', which='major', width=3, length=8)
        # ax.tick_params(axis='y', which='minor', width=2, length=4)

        ax.set_xticks([0, 0.4, 0.8, 1.2, 1.6, 2])
        ax.set_xticklabels(['0', '0.4', '0.8', '1.2', '1.6', '2'], fontsize=33, weight='bold', family='Times New Roman')
        ax.set_yticks([0, 0.4, 0.8, 1.2, 1.6, 2])
        ax.set_yticklabels(['', '0.4', '0.8', '1.2', '1.6', '2'], fontsize=33, weight='bold', family='Times New Roman')

        cbar.ax.tick_params(width=3, length=8)
        ax.set_xticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
        ax.set_yticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
        ax.tick_params(direction='in', which='both')
        ax.tick_params(axis='x', which='major', width=3, length=8)
        ax.tick_params(axis='x', which='minor', width=2, length=4)
        ax.tick_params(axis='y', which='major', width=3, length=8)
        ax.tick_params(axis='y', which='minor', width=2, length=4)


    # plt.rcParams['figure.figsize']=(6,4)
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    # plt.gcf().subplots_adjust(left=0.05, top=0.91, bottom=0.09)
    plt.tight_layout()
    #
    file_path = os.path.join(s, "Xh_Xo_sth.png")
    fig.savefig(file_path)
    plt.show()
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.close('all')



    return Xh,Xo,Z

def coSTH_pu(VLC,xh,xo,s,t1):
    if not os.path.exists(s):
        os.makedirs(s)
    workbook1 = xlrd.open_workbook(r'D:\down\1.xls')  # 文件路径读文件
    names1 = workbook1.sheet_names()
    # print(names1)
    objsth = workbook1.sheet_by_name("Sheet1")
    Xh = np.linspace(0, 2, 201)
    Xo = np.linspace(0, 2, 201)
    # VLC = 0
    xh = round(xh,2)
    xo = round(xo,2)

    xh1 = xh - (t1 * 0.059)
    xo1 = xo + (t1 * 0.059)

    xh1 = round(xh1, 2)
    xo1 = round(xo1, 2)


    X , Y = np.meshgrid(Xh, Xo)

    Eg = X + Y + 1.23 - VLC

    l = objsth.col_values(2)
    f = objsth.col_values(3)
    h = objsth.col_values(4)
    n = objsth.col_values(5)
    # 第2行开始
    l.pop(0)
    f.pop(0)
    h.pop(0)
    n.pop(0)
    Emax = np.max(l)
    emax = np.max(h)
    # 求最小梯度
    DEmin = np.min(np.gradient(l))
    dEmin = np.min(np.gradient(h))
    r = []
    Z = []
    max = 0
    min = 1000
    ans = 0
    ans1 = 0
    for i, Eg0 in zip(Xo, Eg):
        # ans = 0
        for j, Eg1 in zip(Xh, Eg0):
            # print(i,j,Eg1)
            if Eg1 < 0:
                r.append(np.nan)
                continue
            if j >= 0.2 and i >= 0.6:
                if Eg1 < 0.31:
                    Eg1 = 0.31
                E = Eg1
            elif j < 0.2 and i >= 0.6:
                E = Eg1 + 0.2 - j
                if E < 0.31:
                    E = 0.31
                if Eg1 < 0.31:
                    Eg1 = 0.31
            elif j >= 0.2 and i < 0.6:
                E = Eg1 + 0.6 - i
                if E < 0.31:
                    E = 0.31
                if Eg1 < 0.31:
                    Eg1 = 0.31
            elif j < 0.2 and i < 0.6:
                E = Eg1 + 0.8 - i - j
                if E < 0.31:
                    E = 0.31
                if Eg1 < 0.31:
                    Eg1 = 0.31
            # 插值积分

            Eintp = np.arange(Eg1, DEmin + Emax, DEmin)  # 无除
            eintp = np.arange(E, emax + dEmin, dEmin)  # 有除
            Jintp = np.interp(Eintp, l, f)
            jintp = np.interp(eintp, h, n)
            fintp = np.interp(Eintp, h, n)
            Egg = np.trapz(Jintp, Eintp)
            Eh = np.trapz(jintp, eintp)
            Ef = np.trapz(fintp, Eintp)

            nabs = Egg / 1000.37
            ncu = Eh * 1.23 / Egg
            STH = nabs * ncu
            correctedSTH = STH * 1000.37 / (1000.37 + VLC * Ef)
            if xh == j and xo == i:
                ans = correctedSTH * 100
            if xh1 == j and xo1 == i:
                ans1 = correctedSTH * 100
            if correctedSTH >= max:
                max = correctedSTH
            if correctedSTH < min:
                min = correctedSTH
            r.append(correctedSTH * 100)
        Z.append(r)
        r = []

    # print(max)
    # print(min)
    # for i in Z:
    #     print(i)
    # a = 10
    # b = 8   "#86190d",,"#0c08ed"
    max = max * 100
    min = min * 100
    fig, ax = plt.subplots()
    # plt.figure(figsize=(12, 8))
    fig.set_size_inches(10, 8)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    colors = ["#86190d", "#dc0105", "#ffad0d", "#edfc1b", "#8efd72", "#1ffee1", "#03bffe", "#0c08ed"]
    # colors = ["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#da0300","#7e0004"]
    colors.reverse()
    cmap = LinearSegmentedColormap.from_list("custom_rainbow", colors)
    cmap_reversed = cmap.reversed()
    norm = mpl.colors.Normalize(vmin=min, vmax=max)
    # ax.rc('font', size=12, family='serif')
    mpl.rc('font', size=33, family='Times New Roman', weight='bold')
    # 刻度线值, manual=manual_locations
    # manual_locations = [(0.23, 0.64), (0.3, 0.72), (0.42, 0.8), (0.5, 0.83), (0.8, 0.85)]
    C = ax.contour(X, Y, Z, colors='black', linewidths=0)

    cs = ax.contourf(X, Y, Z, 5, cmap=cmap)  # 画出等高图

    x_val, y_val = np.where(np.isclose(Z, max))


    # 在图中标记Z值为0.5的点
    # plt.scatter(xh, xo, color='blue')  # 用红色标出你要标记的点
    # title = ["pH", '\u03c7h', '\u03c7o', "\u03B7abs", "\u03B7cu", "\u03B7STH", '\u03B7\u2032STH']


    # print(x_val)
    # print(y_val)
    #  = X[x_val[0], y_val[0]]]
    # [Y[x_val[0], y_val[0]]

    # ax.plot([X[x_val[0], y_val[0]] , X[x_val[-1],y_val[-1]], Y[x_val[-1], y_val[-1]],Y[x_val[-1], y_val[-1]]]
    #         , color='black', linestyle='-')
    # ax.plot([0, 0.2], [0.6, 0.6], color='black', linestyle='-')
    # ax.plot([0.2, 0.2], [0, 0.6], color='black', linestyle='-')
    # ax.plot([xh1, xh], [xo1, xo], color='black', linestyle='-')
    # ax.plot([xh1, xh], [xo1, xo], color='black', linestyle='-')
    # ax.plot([xh1, xh], [xo1, xo], color='black', linestyle='-')



    # #添加colorbar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)  # 对colorbar的大小进行设置

    cbar.ax.invert_yaxis()

    cbar.update_ticks()  # 显示colorbar的刻度值


    font3 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 33,
             }
    ax.set_xlabel('$\chi$(H$_2$) (eV)', font3)
    ax.set_ylabel('$\chi$(O$_2$) (eV)', font3)
    # 设置主刻度的标签， 带入主刻度旋转角度和字体大小参数
    ax.set_xticks([0, 0.4,  0.8, 1.2,1.6,2])
    ax.set_xticklabels(['0', '0.4', '0.8','1.2', '1.6', '2'], fontsize=30, weight='bold', family='Times New Roman')
    ax.set_yticks([0,  0.4,  0.8,1.2,1.6,2])
    ax.set_yticklabels(['', '0.4', '0.8','1.2','1.6', '2'], fontsize=30, weight='bold', family='Times New Roman')

    cbar.ax.tick_params(width=3, length=8)
    ax.set_xticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
    ax.set_yticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
    ax.tick_params(direction='in', which='both')
    ax.tick_params(axis='x', which='major', width=3, length=8)
    ax.tick_params(axis='x', which='minor', width=2, length=4)
    ax.tick_params(axis='y', which='major', width=3, length=8)
    ax.tick_params(axis='y', which='minor', width=2, length=4)
    # plt.rcParams['figure.figsize']=(6,4)
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    # plt.gcf().subplots_adjust(left=0.05, top=0.91, bottom=0.09)
    plt.tight_layout()
    #
    file_path = os.path.join(s, "Xh_Xo_sth_pu.png")
    fig.savefig(file_path)
    # plt.show()
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.close('all')



    return Xh,Xo,Z

    pass
def CBM_VBM(VLC,C,V,s,t1):
    if not os.path.exists(s):
        os.makedirs(s)
    workbook1 = xlrd.open_workbook(r'D:\down\1.xls')  # 文件路径读文件
    names1 = workbook1.sheet_names()
    C = round(C, 2)
    V = round(V, 2)

    # print(names1)
    objsth = workbook1.sheet_by_name("Sheet1")
    # Xh = np.linspace(0, 1, 101)
    # Xo = np.linspace(0, 1, 101)
    # VLC = 0

    # Y, X = np.meshgrid(Xh, Xo)
    #
    # Eg = X + Y +1.23 - VLC

    l = objsth.col_values(2)
    f = objsth.col_values(3)
    h = objsth.col_values(4)
    n = objsth.col_values(5)
    # 第2行开始
    l.pop(0)
    f.pop(0)
    h.pop(0)
    n.pop(0)

    emax = np.max(h)
    # 求最小梯度
    DEmin = np.min(np.gradient(l))
    dEmin = np.min(np.gradient(h))
    r = []
    Z = []
    max = 0
    min = 1000
    CBM = np.arange(-4.44, -3.23, 0.01)
    VBM = np.arange(-7.3, -5.66, 0.01)
    X, Y = np.meshgrid(CBM, VBM)
    Eg = X - Y
    ans = 0
    for i, Eg0 in zip(VBM, Eg):
        for j, Eg1 in zip(CBM, Eg0):
            # print(i,j,Eg1)
            xh = j + 4.44 + VLC - (t1*0.059)
            xo = -5.67 - i + (t1*0.059)
            if (xh < 0 or xo < 0) or Eg1 < 1.23:
                r.append(np.nan)
                continue
            if xh >= 0.2 and xo >= 0.6:
                E = Eg1

            elif xh < 0.2 and xo >= 0.6:
                E = Eg1 + 0.2 - xh

            elif xh >= 0.2 and xo < 0.6:
                E = Eg1 + 0.6 - xo

            elif xh < 0.2 and xo < 0.6:
                E = Eg1 + 0.8 - xh - xo

            Eintp = np.arange(Eg1, DEmin + emax, DEmin)  # 无除
            eintp = np.arange(E, emax + dEmin, dEmin)  # 有除
            Jintp = np.interp(Eintp, l, f)
            jintp = np.interp(eintp, h, n)
            fintp = np.interp(Eintp, h, n)
            Egg = np.trapz(Jintp, Eintp)
            Eh = np.trapz(jintp, eintp)
            Ef = np.trapz(fintp, Eintp)

            nabs = Egg / 1000.37

            ncu = Eh * 1.23 / Egg
            # STH = Eh * 1.23 / 1000.37
            STH = nabs*ncu
            correctedSTH = STH*1000.37/(1000.37+VLC*Ef)
            if C == round(j,2) and V == round(i,2):
                ans = correctedSTH * 100
            if correctedSTH >= max:
                max = correctedSTH
            if correctedSTH < min:
                min = correctedSTH

            r.append(correctedSTH * 100)
        Z.append(r)
        r = []


    max = max * 100
    min = min * 100
    print(max)
    print(min)
    fig, ax = plt.subplots(figsize=(10, 8))
    # plt.figure(figsize=(10, 8))
    fig.set_size_inches(12, 10)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    colors = ["#86190d", "#dc0105", "#ffad0d", "#edfc1b", "#8efd72", "#1ffee1", "#03bffe", "#0c08ed"]
    # colors = ["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#da0300","#7e0004"]
    colors.reverse()
    cmap = LinearSegmentedColormap.from_list("custom_rainbow", colors)
    cmap_reversed = cmap.reversed()
    norm = mpl.colors.Normalize(vmin=min, vmax=max)
    # ax.rc('font', size=12, family='serif')
    mpl.rc('font', size=40, family='Times New Roman', weight='bold')
    cs = ax.contourf(X, Y, Z, cmap=cmap)  # 画出等高图

    x_val, y_val = np.where(np.isclose(Z, max))
    # print(x_val)
    # print(y_val)

    # 在图中标记Z值为0.5的点\u2032
    # title = ["pH", '\u03c7h', '\u03c7o', "\u03B7abs", "\u03B7cu", "\u03B7STH", '\u03B7\u2032STH']
    # plt.annotate(f'({C:.2f}, {V:.2f})', (C, V), textcoords="offset points", xytext=(0, -20),#坐标
    #              ha='center', fontsize=20)
    if VLC == 0:
        plt.annotate(f'$\eta_{{STH}}={ans:.2f}\%$', (C, V), textcoords="offset points", xytext=(0, 10),
                 ha='center', fontsize=25)
    #
    else:
        plt.annotate(f'{ans:.2f}%', (C, V), textcoords="offset points", xytext=(30, 10),
                     ha='center', fontsize=25)
    if VLC == 0:
        for i in range(len(x_val)):
            ax.scatter(X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]], marker='*', c='black',s = 1)
        plt.annotate(f'$\eta^{{max}}_{{STH}}={max:.2f}\%$', (-4.3, -6.4),
                     textcoords="offset points", xytext=(60, -10), ha='center', fontsize=25, c='black')
    else:
        for i in range(len(x_val)):
            ax.scatter(X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]], marker='*', c='black',s = 100)
        # plt.annotate(f'$\eta^{{max}}_{{STH}}={max:.2f}\%$', (X[x_val[-1], y_val[-1]], Y[x_val[-1], y_val[-1]]),
        #              textcoords="offset points", xytext=(60, 20), ha='center', fontsize=25, c='black')
        # plt.annotate('*', (X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]]), fontsize=20)

    if C >= -4.44 and C<-2.23 and V>=-7.3 and V<-4.66:
        ax.scatter(C, V, marker='*',c = 'black',s = 100)
        plt.annotate(f'pH={0}', (C, V), textcoords="offset points", xytext=(0, -30),
                     ha='center', fontsize=25)





    # ax.colorbars(label='correctSTH')
    # #添加colorbar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)  # 对colorbar的大小进行设置
    # ticks = [1, 3, 6, 9, 12, 15]  # 刻度值，包括最大值
    # tick_labels = ['1', '3', '6', '9', '12', '15']  # 对应的刻度标签
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels(tick_labels)
    #
    # # 使用FixedLocator和FixedFormatter
    # cbar.locator = FixedLocator(ticks)
    # cbar.formatter = FixedFormatter(tick_labels)
    cbar.ax.invert_yaxis()
    # cbar.set_ticks([10,15,20,25,30,35,max])
    # cbar.set_ticklabels(['10', '15', '20', '25', '30','35','38.18'])
    # cbar.ax.set_title('η’$_{STH}$ $\%$',fontsize=40,weight='bold', family='Times New Roman',pad=15)
    cbar.update_ticks()  # 显示colorbar的刻度值
    # ticks = cbar.get_ticks()  # 获取当前的刻度标签
    # new_ticks = ticks +0  # 向上移动2个单位
    # cbar.set_ticks(new_ticks)  # 设置新的刻度标签

    font3 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 40,
             }
    ax.set_xlabel('CBM (eV)', font3)
    ax.set_ylabel('VBM (eV)', font3)
    # 设置主刻度的标签， 带入主刻度旋转角度和字体大小参数
    ax.set_xticks([-4.44, -4.24, -4.04, -3.84, -3.64, -3.44, -3.23])
    ax.set_xticklabels(['-4.44', '-4.24', '-4.04', '-3.84', '-3.64', '-3.44', '-3.24'], fontsize=35, weight='bold',
                       family='Times New Roman')
    ax.set_yticks([-7.47, -7.17 ,-6.87, -6.57, -6.27, -5.97, -5.67])
    ax.set_yticklabels(['-7.47','-7.17', '-6.87', '-6.57', '-6.27', '-5.97', '-5.67'], fontsize=35, weight='bold',
                       family='Times New Roman')

    cbar.ax.tick_params(width=3, length=8)
    # ax.set_xticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
    # ax.set_yticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
    ax.tick_params(direction='in', which='both')
    ax.tick_params(axis='x', which='major', width=3, length=8)
    # ax.tick_params(axis='x', which='minor', width=2, length=4)
    ax.tick_params(axis='y', which='major', width=3, length=8)
    # ax.tick_params(axis='y', which='minor', width=2, length=4)
    # plt.rcParams['figure.figsize']=(6,4)
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    # plt.gcf().subplots_adjust(left=0.05, top=0.91, bottom=0.09)

    plt.tight_layout()
    file_path = os.path.join(s, "CBM_VBM_sth.png")
    fig.savefig(file_path)
    #
    plt.show()
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.close('all')
    # fig.savefig(r'D:\tu\python\CBM_VBM_0.png')
    return CBM,VBM,Z

def CBM_VBM_J(VLC,C,V,s,t1):
    if not os.path.exists(s):
        os.makedirs(s)
    workbook1 = xlrd.open_workbook(r'D:\down\1.xls')  # 文件路径读文件
    names1 = workbook1.sheet_names()
    C = round(C, 2)
    V = round(V, 2)

    # print(names1)
    objsth = workbook1.sheet_by_name("Sheet1")
    # Xh = np.linspace(0, 1, 101)
    # Xo = np.linspace(0, 1, 101)
    # VLC = 0

    # Y, X = np.meshgrid(Xh, Xo)
    #
    # Eg = X + Y +1.23 - VLC

    l = objsth.col_values(2)
    f = objsth.col_values(3)
    h = objsth.col_values(4)
    n = objsth.col_values(5)
    # 第2行开始
    l.pop(0)
    f.pop(0)
    h.pop(0)
    n.pop(0)

    emax = np.max(h)
    # 求最小梯度
    DEmin = np.min(np.gradient(l))
    dEmin = np.min(np.gradient(h))
    r = []
    Z = []
    max = 0
    min = 1000
    CBM = np.arange(-5.03, -3.22, 0.01)
    VBM = np.arange(-7.66, -5.66, 0.01)
    X, Y = np.meshgrid(CBM, VBM)
    Eg = X - Y
    ans = 0
    for i, Eg0 in zip(VBM, Eg):
        for j, Eg1 in zip(CBM, Eg0):
            # print(i,j,Eg1)
            xh = j + 4.44 + VLC - (t1*0.059)
            xo = -5.67 - i + (t1*0.059)
            if (xh < 0 or xo < 0):
                r.append(np.nan)
                continue
            if xh >= 0.2 and xo >= 0.6:
                if Eg1<0.31:
                    Eg1 = 0.31
                E = Eg1

            elif xh < 0.2 and xo >= 0.6:
                if Eg1<0.31:
                    Eg1 = 0.31
                E = Eg1 + 0.2 - xh
                if E<0.31:
                    E = 0.31

            elif xh >= 0.2 and xo < 0.6:
                if Eg1<0.31:
                    Eg1 = 0.31
                E = Eg1 + 0.6 - xo
                if E<0.31:
                    E = 0.31
            elif xh < 0.2 and xo < 0.6:
                if Eg1<0.31:
                    Eg1 = 0.31
                E = Eg1 + 0.8 - xh - xo
                if E<0.31:
                    E = 0.31

            Eintp = np.arange(Eg1, DEmin + emax, DEmin)  # 无除
            eintp = np.arange(E, emax + dEmin, dEmin)  # 有除
            Jintp = np.interp(Eintp, l, f)
            jintp = np.interp(eintp, h, n)
            fintp = np.interp(Eintp, h, n)
            Egg = np.trapz(Jintp, Eintp)
            Eh = np.trapz(jintp, eintp)
            Ef = np.trapz(fintp, Eintp)

            nabs = Egg / 1000.37

            ncu = Eh * 1.23 / Egg
            # STH = Eh * 1.23 / 1000.37
            STH = nabs*ncu
            correctedSTH = STH*1000.37/(1000.37+VLC*Ef)
            if C == round(j,2) and V == round(i,2):
                ans = correctedSTH * 100

            if correctedSTH >= max:
                max = correctedSTH
            if correctedSTH < min:
                min = correctedSTH

            r.append(correctedSTH * 100)
        Z.append(r)
        r = []


    max = max * 100
    min = min * 100
    print(max)
    print(min)
    fig, ax = plt.subplots(figsize=(10, 8))
    # plt.figure(figsize=(10, 8))
    fig.set_size_inches(10, 8)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    colors = ["#86190d", "#dc0105", "#ffad0d", "#edfc1b", "#8efd72", "#1ffee1", "#03bffe", "#0c08ed"]
    # colors = ["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#da0300","#7e0004"]
    colors.reverse()
    cmap = LinearSegmentedColormap.from_list("custom_rainbow", colors)
    cmap_reversed = cmap.reversed()
    norm = mpl.colors.Normalize(vmin=min, vmax=max)
    # ax.rc('font', size=12, family='serif')
    mpl.rc('font', size=33, family='Times New Roman', weight='bold')
    cs = ax.contourf(X, Y, Z, cmap=cmap)  # 画出等高图

    x_val, y_val = np.where(np.isclose(Z, max))
    # print(x_val)
    # print(y_val)

    # 在图中标记Z值为0.5的点\u2032
    # title = ["pH", '\u03c7h', '\u03c7o', "\u03B7abs", "\u03B7cu", "\u03B7STH", '\u03B7\u2032STH']
    # plt.annotate(f'({C:.2f}, {V:.2f})', (C, V), textcoords="offset points", xytext=(0, -20),#坐标
    #              ha='center', fontsize=20)
    # if VLC == 0:
    #     plt.annotate(f'{ans:.2f}%', (C, V), textcoords="offset points", xytext=(30, 10),
    #              ha='center', fontsize=15)
    #
    # else:
    #     plt.annotate(f'{ans:.2f}%', (C, V), textcoords="offset points", xytext=(30, 10),
    #                  ha='center', fontsize=15)
    if VLC == 0:
        for i in range(len(x_val)):
            ax.scatter(X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]], marker='*', c='black',s = 1)
        plt.annotate(f'$\eta^{{max}}_{{STH}}={max:.2f}\%$', (-4.3, -6.4),
                     textcoords="offset points", xytext=(60, -10), ha='center', fontsize=25, c='black')
    else:
        for i in range(len(x_val)):
            ax.scatter(X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]], marker='*', c='white',s = 300)
        # plt.annotate(f'$\eta^{{max}}_{{STH}}={max:.2f}\%$', (X[x_val[-1], y_val[-1]], Y[x_val[-1], y_val[-1]]),
        #              textcoords="offset points", xytext=(120, -50), ha='center', fontsize=30, c='black')
        # plt.annotate('*', (X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]]), fontsize=20)



    if C >= -6 and C<-3.23 and V>=-7.3 and V<-4.66:
        ax.scatter(C, V, marker='*',c = 'black', s=300)
        # plt.annotate(f'pH={t1}', (C, V), textcoords="offset points", xytext=(0, -30),
        #              ha='center', fontsize=30)
        # plt.annotate(f'$\eta^\prime_{{STH}}={ans:.2f}\%$', (C,V),
        #              textcoords="offset points", xytext=(60, 20), ha='center', fontsize=30, c='black')
    #
    # ax.colorbars(label='correctSTH')
    # #添加colorbar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)  # 对colorbar的大小进行设置
    # ticks = [1, 3, 6, 9, 12, 15]  # 刻度值，包括最大值
    # tick_labels = ['1', '3', '6', '9', '12', '15']  # 对应的刻度标签
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels(tick_labels)
    #
    # # 使用FixedLocator和FixedFormatter
    # cbar.locator = FixedLocator(ticks)
    # cbar.formatter = FixedFormatter(tick_labels)
    cbar.ax.invert_yaxis()
    # cbar.set_ticks([10,15,20,25,30,35,max])
    # cbar.set_ticklabels(['10', '15', '20', '25', '30','35','38.18'])
    # cbar.ax.set_title('η’$_{STH}$ $\%$',fontsize=40,weight='bold', family='Times New Roman',pad=15)
    cbar.update_ticks()  # 显示colorbar的刻度值
    # ticks = cbar.get_ticks()  # 获取当前的刻度标签
    # new_ticks = ticks +0  # 向上移动2个单位
    # cbar.set_ticks(new_ticks)  # 设置新的刻度标签

    font3 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 33,
             }
    ax.set_xlabel('CBM (eV)', font3)
    ax.set_ylabel('VBM (eV)', font3)
    # 设置主刻度的标签， 带入主刻度旋转角度和字体大小参数
    ax.set_xticks([ -5.03, -4.53, -4.03, -3.63, -3.23])
    ax.set_xticklabels([ '-5.03', '-4.53', '-4.03', '-3.63', '-3.23'], fontsize=33, weight='bold',
                       family='Times New Roman')
    ax.set_yticks([ -7.47,-7.17 ,-6.87, -6.57, -6.27, -5.97, -5.67])
    ax.set_yticklabels(['-7.47','-7.17', '-6.87', '-6.57', '-6.27', '-5.97', '-5.67'], fontsize=33, weight='bold',
                       family='Times New Roman')

    cbar.ax.tick_params(width=3, length=8)
    # ax.set_xticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
    # ax.set_yticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
    ax.tick_params(direction='in', which='both')
    ax.tick_params(axis='x', which='major', width=3, length=8)
    # ax.tick_params(axis='x', which='minor', width=2, length=4)
    ax.tick_params(axis='y', which='major', width=3, length=8)
    # ax.tick_params(axis='y', which='minor', width=2, length=4)
    # plt.rcParams['figure.figsize']=(6,4)
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    # plt.gcf().subplots_adjust(left=0.05, top=0.91, bottom=0.09)

    plt.tight_layout()
    file_path = os.path.join(s, "CBM_VBM_J_sth.png")
    fig.savefig(file_path)
    #
    plt.show()
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.close('all')
    # fig.savefig(r'D:\tu\python\CBM_VBM_0.png')
    return CBM,VBM,Z

def CBM_VBM_pu(VLC,C,V,s,t1):
    if not os.path.exists(s):
        os.makedirs(s)
    workbook1 = xlrd.open_workbook(r'D:\down\1.xls')  # 文件路径读文件
    names1 = workbook1.sheet_names()
    C = round(C, 2)
    V = round(V, 2)

    # print(names1)
    objsth = workbook1.sheet_by_name("Sheet1")
    # Xh = np.linspace(0, 1, 101)
    # Xo = np.linspace(0, 1, 101)
    # VLC = 0

    # Y, X = np.meshgrid(Xh, Xo)
    #
    # Eg = X + Y +1.23 - VLC

    l = objsth.col_values(2)
    f = objsth.col_values(3)
    h = objsth.col_values(4)
    n = objsth.col_values(5)
    # 第2行开始
    l.pop(0)
    f.pop(0)
    h.pop(0)
    n.pop(0)

    emax = np.max(h)
    # 求最小梯度
    DEmin = np.min(np.gradient(l))
    dEmin = np.min(np.gradient(h))
    r = []
    Z = []
    max = 0
    min = 1000
    CBM = np.arange(-4.44, -3.23, 0.01)
    VBM = np.arange(-7.3, -5.66, 0.01)
    X, Y = np.meshgrid(CBM, VBM)
    Eg = X - Y
    ans = 0
    for i, Eg0 in zip(VBM, Eg):
        for j, Eg1 in zip(CBM, Eg0):
            # print(i,j,Eg1)
            xh = j + 4.44 + VLC
            xo = -5.67 - i
            if (xh < 0 or xo < 0) or Eg1 < 1.23:
                r.append(np.nan)
                continue
            if xh >= 0.2 and xo >= 0.6:
                E = Eg1

            elif xh < 0.2 and xo >= 0.6:
                E = Eg1 + 0.2 - xh

            elif xh >= 0.2 and xo < 0.6:
                E = Eg1 + 0.6 - xo

            elif xh < 0.2 and xo < 0.6:
                E = Eg1 + 0.8 - xh - xo

            Eintp = np.arange(Eg1, DEmin + emax, DEmin)  # 无除
            eintp = np.arange(E, emax + dEmin, dEmin)  # 有除
            Jintp = np.interp(Eintp, l, f)
            jintp = np.interp(eintp, h, n)
            fintp = np.interp(Eintp, h, n)
            Egg = np.trapz(Jintp, Eintp)
            Eh = np.trapz(jintp, eintp)
            Ef = np.trapz(fintp, Eintp)

            nabs = Egg / 1000.37

            ncu = Eh * 1.23 / Egg
            # STH = Eh * 1.23 / 1000.37
            STH = nabs*ncu
            correctedSTH = STH*1000.37/(1000.37+VLC*Ef)
            if C == round(j,2) and V == round(i,2):
                ans = correctedSTH * 100
            if correctedSTH >= max:
                max = correctedSTH
            if correctedSTH < min:
                min = correctedSTH

            r.append(correctedSTH * 100)
        Z.append(r)
        r = []


    max = max * 100
    min = min * 100
    # print(max)
    # print(min)
    fig, ax = plt.subplots(figsize=(10, 8))
    # plt.figure(figsize=(10, 8))
    fig.set_size_inches(10, 8)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    colors = ["#86190d", "#dc0105", "#ffad0d", "#edfc1b", "#8efd72", "#1ffee1", "#03bffe", "#0c08ed"]
    # colors = ["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#da0300","#7e0004"]
    colors.reverse()
    cmap = LinearSegmentedColormap.from_list("custom_rainbow", colors)
    cmap_reversed = cmap.reversed()
    norm = mpl.colors.Normalize(vmin=min, vmax=max)
    # ax.rc('font', size=12, family='serif')
    mpl.rc('font', size=33, family='Times New Roman', weight='bold')
    cs = ax.contourf(X, Y, Z, cmap=cmap)  # 画出等高图

    x_val, y_val = np.where(np.isclose(Z, max))
    # print(x_val)
    # print(y_val)

    # 在图中标记Z值为0.5的点\u2032
    # title = ["pH", '\u03c7h', '\u03c7o', "\u03B7abs", "\u03B7cu", "\u03B7STH", '\u03B7\u2032STH']
    # plt.annotate(f'({C:.2f}, {V:.2f})', (C, V), textcoords="offset points", xytext=(0, -20),#坐标
    #              ha='center', fontsize=20)

    #
    # ax.colorbars(label='correctSTH')
    # #添加colorbar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)  # 对colorbar的大小进行设置
    # ticks = [1, 3, 6, 9, 12, 15]  # 刻度值，包括最大值
    # tick_labels = ['1', '3', '6', '9', '12', '15']  # 对应的刻度标签
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels(tick_labels)
    #
    # # 使用FixedLocator和FixedFormatter
    # cbar.locator = FixedLocator(ticks)
    # cbar.formatter = FixedFormatter(tick_labels)
    cbar.ax.invert_yaxis()
    # cbar.set_ticks([10,15,20,25,30,35,max])
    # cbar.set_ticklabels(['10', '15', '20', '25', '30','35','38.18'])
    # cbar.ax.set_title('η’$_{STH}$ $\%$',fontsize=40,weight='bold', family='Times New Roman',pad=15)
    cbar.update_ticks()  # 显示colorbar的刻度值
    # ticks = cbar.get_ticks()  # 获取当前的刻度标签
    # new_ticks = ticks +0  # 向上移动2个单位
    # cbar.set_ticks(new_ticks)  # 设置新的刻度标签

    font3 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 33,
             }
    ax.set_xlabel('CBM (eV)', font3)
    ax.set_ylabel('VBM (eV)' , font3)
    # 设置主刻度的标签， 带入主刻度旋转角度和字体大小参数
    ax.set_xticks([-4.44, -4.24, -4.04, -3.84, -3.64, -3.44, -3.23])
    ax.set_xticklabels(['-4.44', '-4.24', '-4.04', '-3.84', '-3.64', '-3.43', '-3.24'], fontsize=24, weight='bold',
                       family='Times New Roman')
    ax.set_yticks([ -7 ,-6.7, -6.4, -6.1, -5.9, -5.67])
    ax.set_yticklabels(['-7', '-6.7', '-6.4', '-6.1', '-5.9', '-5.67'], fontsize=24, weight='bold',
                       family='Times New Roman')

    cbar.ax.tick_params(width=3, length=8)
    ax.set_xticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
    ax.set_yticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
    ax.tick_params(direction='in', which='both')
    ax.tick_params(axis='x', which='major', width=3, length=8)
    ax.tick_params(axis='x', which='minor', width=2, length=4)
    ax.tick_params(axis='y', which='major', width=3, length=8)
    ax.tick_params(axis='y', which='minor', width=2, length=4)
    # plt.rcParams['figure.figsize']=(6,4)
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    # plt.gcf().subplots_adjust(left=0.05, top=0.91, bottom=0.09)

    plt.tight_layout()
    file_path = os.path.join(s, "CBM_VBM_sth_pu.png")
    fig.savefig(file_path)
    #
    # plt.show()
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.close('all')
    # fig.savefig(r'D:\tu\python\CBM_VBM_0.png')
    return CBM,VBM,Z



    pass
def coSTH_Z(VLC,xh,xo,s,E1,EA,EB):
    if not os.path.exists(s):
        os.makedirs(s)
    # print(len(E))
    len1 = int(len(E1))
    workbook1 = xlrd.open_workbook(r'D:\down\1.xls')  # 文件路径读文件
    names1 = workbook1.sheet_names()
    # print(names1)
    objsth = workbook1.sheet_by_name("Sheet1")
    Xh = np.linspace(0, 2, 201)
    Xo = np.linspace(0, 2, 201)
    # VLC = 0
    xh = round(xh, 2)
    xo = round(xo, 2)
    t1 = E1[-1]
    xh1 = xh - (t1 * 0.059)
    xo1 = xo + (t1 * 0.059)

    xh1 = round(xh1, 2)
    xo1 = round(xo1, 2)
    # VLC = 0

    X, Y = np.meshgrid(Xh, Xo)

    Eg = X + Y + 1.23 - VLC

    l = objsth.col_values(2)
    f = objsth.col_values(3)
    h = objsth.col_values(4)
    n = objsth.col_values(5)
    # 第2行开始
    l.pop(0)
    f.pop(0)
    h.pop(0)
    n.pop(0)
    Emax = np.max(l)
    emax = np.max(h)
    # 求最小梯度
    DEmin = np.min(np.gradient(l))
    dEmin = np.min(np.gradient(h))
    r = []
    Z = []
    max1 = 0
    min1 = 1000
    ans = 0
    ans1 = 0
    for i, Eg0 in zip(Xo, Eg):

        for j, Eg1 in zip(Xh, Eg0):
            # print(i,j,Eg1)
            E = max(EA,EB)
            # 插值积分
            Eintp = np.arange(Eg1, DEmin + Emax, DEmin)  # 无除
            eintp = np.arange(E, emax + dEmin, dEmin)  # 有除
            Jintp = np.interp(Eintp, l, f)
            jintp = np.interp(eintp, h, n)
            fintp = np.interp(Eintp, h, n)
            Egg = np.trapz(Jintp, Eintp)
            Eh = np.trapz(jintp, eintp)
            Ef = np.trapz(fintp, Eintp)
            nabs = Egg / 1000.37
            ncu = Eh * 1.23 / Egg/2
            STH = nabs * ncu
            correctedSTH = STH * 1000.37 / (1000.37 + VLC * Ef)
            if xh == round(j,2) and xo == round(i,2):
                ans = correctedSTH * 100
            if xh1 == round(j,2) and xo1 == round(i,2):
                ans1 = correctedSTH * 100
            if correctedSTH >= max1:
                max1 = correctedSTH
            if correctedSTH < min1:
                min1 = correctedSTH

            r.append(correctedSTH * 100)
        Z.append(r)
        r = []

    max1 = max1 * 100
    min1 = min1 * 100
    print(max1)
    fig, ax = plt.subplots()
    # plt.figure(figsize=(12, 8))
    fig.set_size_inches(10, 8)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    colors = ["#86190d", "#dc0105", "#ffad0d", "#edfc1b", "#8efd72", "#1ffee1", "#03bffe", "#0c08ed"]
    # colors = ["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#da0300","#7e0004"]
    colors.reverse()
    cmap = LinearSegmentedColormap.from_list("custom_rainbow", colors)
    cmap_reversed = cmap.reversed()
    norm = mpl.colors.Normalize(vmin=min1, vmax=max1)
    # ax.rc('font', size=12, family='serif')
    mpl.rc('font', size=33, family='Times New Roman', weight='bold')
    # 刻度线值, manual=manual_locations
    # manual_locations = [(0.23, 0.64), (0.3, 0.72), (0.42, 0.8), (0.5, 0.83), (0.8, 0.85)]
    C = ax.contour(X, Y, Z, colors='black', linewidths=0)

    cs = ax.contourf(X, Y, Z, 5, cmap=cmap)  # 画出等高图

    x_val, y_val = np.where(np.isclose(Z, max1))

    # 在图中标记Z值为0.5的点
    # plt.scatter(xh, xo, color='blue')  # 用红色标出你要标记的点
    # title = ["pH", '\u03c7h', '\u03c7o', "\u03B7abs", "\u03B7cu", "\u03B7STH", '\u03B7\u2032STH']

    # if VLC == 0:
    #     plt.annotate(f'{ans:.2f}%', (xh, xo), textcoords="offset points", xytext=(65, 10),
    #                  ha='center', fontsize=15)
    #     plt.annotate(f'{ans1:.2f}%', (xh1, xo1), textcoords="offset points", xytext=(30, 30),
    #                  ha='center', fontsize=15)
    #     # plt.annotate(f'\u03B7STH = {ans:.2f}%', (xh1, xo1), textcoords="offset points", xytext=(0, 20),
    #     #              ha='center', fontsize=20)
    # else:
    #     plt.annotate(f'{ans:.2f}%', (xh, xo), textcoords="offset points", xytext=(65, 10),
    #                  ha='center', fontsize=15)
    #     plt.annotate(f'{ans1:.2f}%', (xh1, xo1), textcoords="offset points", xytext=(30, 30),
    #                  ha='center', fontsize=15)
        # plt.annotate(f'\u03B7\u2032STH = {ans:.2f}%', (xh, xo), textcoords="offset points", xytext=(0, 20),
        #              ha='center', fontsize=20)
    # print(x_val)
    # print(y_val)
    #  = X[x_val[0], y_val[0]]]
    # [Y[x_val[0], y_val[0]]
    if VLC == 0:
        for i in range(len(x_val)):
            ax.scatter(X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]], marker='\u002E', c='black', s=1)
    else:
        for i in range(len(x_val)):
            ax.scatter(X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]], marker='\u002E', c='black', s=50)

    # ax.plot([X[x_val[0], y_val[0]] , X[x_val[-1],y_val[-1]], Y[x_val[-1], y_val[-1]],Y[x_val[-1], y_val[-1]]]
    #         , color='black', linestyle='-')
    # ax.plot([0, 0.2], [0.6, 0.6], color='black', linestyle='-')
    # ax.plot([0.2, 0.2], [0, 0.6], color='black', linestyle='-')
    # ax.plot([xh1, xh], [xo1, xo], color='black', linestyle='-')
    # ax.plot([xh1, xh], [xo1, xo], color='black', linestyle='-')
    # ax.plot([xh1, xh], [xo1, xo], color='black', linestyle='-')
    if VLC == 0:
        plt.annotate(f'$\eta^{{lim}}_{{STH}}={max1:.2f}\%$', (0.6, 1.6),
                     textcoords="offset points", xytext=(0, 0), ha='center', fontsize=15, c='black')
    else:
        plt.annotate(f'$\eta^{{lim}}_{{STH}}={max1:.2f}\%$', (X[x_val[-1], y_val[-1]], Y[x_val[-1], y_val[-1]]),
                     textcoords="offset points", xytext=(30, 20), ha='center', fontsize=15, c='black')
    # if xh < 2 and xo < 2:
    #     for i in range(len1):
    #         xh_p = xh - (0.059 * i)
    #         xo_p = xo + (0.059 * i)
    #         xh_p = round(xh_p, 2)
    #         xo_p = round(xo_p, 2)
    #         if i == 0:
    #             plt.annotate(f'0', (xh_p, xo_p), textcoords="offset points", xytext=(30, 10),
    #                          ha='center', fontsize=15)
    #             plt.annotate(f'pH', (xh_p, xo_p), textcoords="offset points", xytext=(30, -10),
    #                          ha='center', fontsize=15)
    #         else:
    #             plt.annotate(f'{i}', (xh_p, xo_p), textcoords="offset points", xytext=(30, 10),
    #                          ha='center', fontsize=15)
    #         ax.scatter(xh_p, xo_p, marker='*', c='black')
    # if xh1 < 2 and xo1 < 2:
    #     plt.annotate(f'pH={t1}', (xh1, xo1), textcoords="offset points", xytext=(30, 0),
    #                  ha='center', fontsize=15)
    #     ax.scatter(xh1, xo1, marker='*', c='black')

    # ax.plot([xh1, xh], [xo1, xo], color='black', linestyle='-')

    # #添加colorbar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)  # 对colorbar的大小进行设置

    cbar.ax.invert_yaxis()

    cbar.update_ticks()  # 显示colorbar的刻度值

    font3 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 33,
             }
    ax.set_xlabel('$\chi$(H$_2$) (eV)', font3)
    ax.set_ylabel('$\chi$(O$_2$) (eV)', font3)
    # 设置主刻度的标签， 带入主刻度旋转角度和字体大小参数
    ax.set_xticks([0, 0.4, 0.8, 1.2, 1.6, 2])
    ax.set_xticklabels(['0', '0.4', '0.8', '1.2', '1.6', '2'], fontsize=30, weight='bold', family='Times New Roman')
    ax.set_yticks([0, 0.4, 0.8, 1.2, 1.6, 2])
    ax.set_yticklabels(['', '0.4', '0.8', '1.2', '1.6', '2'], fontsize=30, weight='bold', family='Times New Roman')

    cbar.ax.tick_params(width=3, length=8)
    ax.set_xticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
    ax.set_yticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
    ax.tick_params(direction='in', which='both')
    ax.tick_params(axis='x', which='major', width=3, length=8)
    ax.tick_params(axis='x', which='minor', width=2, length=4)
    ax.tick_params(axis='y', which='major', width=3, length=8)
    ax.tick_params(axis='y', which='minor', width=2, length=4)
    # plt.rcParams['figure.figsize']=(6,4)
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    # plt.gcf().subplots_adjust(left=0.05, top=0.91, bottom=0.09)
    plt.tight_layout()
    file_path = os.path.join(s, "Xh_Xo_sth_Z.png")
    fig.savefig(file_path)
    #
    print(1)
    plt.show()
    mpl.rcParams.update(mpl.rcParamsDefault)
    # fig.savefig(r'D:\tu\python\myplot_0_g.png')
    print(1)
    return Xh,Xo,Z

def coSTH_Z_J(VLC,xh,xo,s,E):
    if not os.path.exists(s):
        os.makedirs(s)
    # print(len(E))
    len1 = int(len(E))
    workbook1 = xlrd.open_workbook(r'D:\down\1.xls')  # 文件路径读文件
    names1 = workbook1.sheet_names()
    # print(names1)
    objsth = workbook1.sheet_by_name("Sheet1")
    Xh = np.linspace(0, 2, 201)
    Xo = np.linspace(0, 2, 201)
    # VLC = 0
    xh = round(xh, 2)
    xo = round(xo, 2)
    t1 = E[-1]
    xh1 = xh - (t1 * 0.059)
    xo1 = xo + (t1 * 0.059)

    xh1 = round(xh1, 2)
    xo1 = round(xo1, 2)
    # VLC = 0

    X, Y = np.meshgrid(Xh, Xo)

    Eg = X + Y + 1.23 - VLC

    l = objsth.col_values(2)
    f = objsth.col_values(3)
    h = objsth.col_values(4)
    n = objsth.col_values(5)
    # 第2行开始
    l.pop(0)
    f.pop(0)
    h.pop(0)
    n.pop(0)
    Emax = np.max(l)
    emax = np.max(h)
    # 求最小梯度
    DEmin = np.min(np.gradient(l))
    dEmin = np.min(np.gradient(h))
    r = []
    Z = []
    max1 = 0
    min1 = 1000
    ans = 0
    ans1 = 0
    for i, Eg0 in zip(Xo, Eg):

        for j, Eg1 in zip(Xh, Eg0):
            # print(i,j,Eg1)
            E1 = 0
            if Eg1 < 0:
                r.append(np.nan)
                continue
            if j >= 0.2 and i >= 0.6:
                if Eg1 < 0.31:
                    Eg1 = 0.31
                E1 = Eg1
            elif j < 0.2 and i >= 0.6:
                E1 = Eg1 + 0.2 - j
                if E1 < 0.31:
                    E1 = 0.31
                if Eg1 < 0.31:
                    Eg1 = 0.31
            elif j >= 0.2 and i < 0.6:
                E1 = Eg1 + 0.6 - i
                if E1 < 0.31:
                    E1 = 0.31
                if Eg1 < 0.31:
                    Eg1 = 0.31
            elif j < 0.2 and i < 0.6:
                E1 = Eg1 + 0.8 - i - j
                if E1 < 0.31:
                    E1 = 0.31
                if Eg1 < 0.31:
                    Eg1 = 0.31
            # 插值积分
            Eintp = np.arange(Eg1, DEmin + Emax, DEmin)  # 无除
            eintp = np.arange(E1, emax + dEmin, dEmin)  # 有除
            Jintp = np.interp(Eintp, l, f)
            jintp = np.interp(eintp, h, n)
            fintp = np.interp(Eintp, h, n)
            Egg = np.trapz(Jintp, Eintp)
            Eh = np.trapz(jintp, eintp)
            Ef = np.trapz(fintp, Eintp)
            nabs = Egg / 1000.37
            ncu = Eh * 1.23 / Egg/2
            STH = nabs * ncu
            correctedSTH = STH * 1000.37 / (1000.37 + VLC * Ef)
            if xh == round(j,2) and xo == round(i,2):
                ans = correctedSTH * 100
            if xh1 == round(j,2) and xo1 == round(i,2):
                ans1 = correctedSTH * 100
            if correctedSTH >= max1:
                max1 = correctedSTH
            if correctedSTH < min1:
                min1 = correctedSTH
            r.append(correctedSTH * 100)
        Z.append(r)
        r = []

    max1 = max1 * 100
    min1 = min1 * 100
    print(max1)
    fig, ax = plt.subplots()
    # plt.figure(figsize=(12, 8))
    fig.set_size_inches(10, 8)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    colors = ["#86190d", "#dc0105", "#ffad0d", "#edfc1b", "#8efd72", "#1ffee1", "#03bffe", "#0c08ed"]
    # colors = ["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#da0300","#7e0004"]
    colors.reverse()
    cmap = LinearSegmentedColormap.from_list("custom_rainbow", colors)
    cmap_reversed = cmap.reversed()
    norm = mpl.colors.Normalize(vmin=min1, vmax=max1)
    # ax.rc('font', size=12, family='serif')
    mpl.rc('font', size=33, family='Times New Roman', weight='bold')
    # 刻度线值, manual=manual_locations
    # manual_locations = [(0.23, 0.64), (0.3, 0.72), (0.42, 0.8), (0.5, 0.83), (0.8, 0.85)]
    C = ax.contour(X, Y, Z, colors='black', linewidths=0)

    cs = ax.contourf(X, Y, Z, 5, cmap=cmap)  # 画出等高图

    x_val, y_val = np.where(np.isclose(Z, max1))

    # 在图中标记Z值为0.5的点
    # plt.scatter(xh, xo, color='blue')  # 用红色标出你要标记的点
    # title = ["pH", '\u03c7h', '\u03c7o', "\u03B7abs", "\u03B7cu", "\u03B7STH", '\u03B7\u2032STH']

    if VLC == 0:
        plt.annotate(f'{ans:.2f}%', (xh, xo), textcoords="offset points", xytext=(65, 10),
                     ha='center', fontsize=15)
        plt.annotate(f'{ans1:.2f}%', (xh1, xo1), textcoords="offset points", xytext=(30, 30),
                     ha='center', fontsize=15)
        # plt.annotate(f'\u03B7STH = {ans:.2f}%', (xh1, xo1), textcoords="offset points", xytext=(0, 20),
        #              ha='center', fontsize=20)
    else:
        plt.annotate(f'{ans:.2f}%', (xh, xo), textcoords="offset points", xytext=(-10, -30),
                     ha='center', fontsize=15)
        plt.annotate(f'{ans1:.2f}%', (xh1, xo1), textcoords="offset points", xytext=(30, 30),
                     ha='center', fontsize=15)
        # plt.annotate(f'\u03B7\u2032STH = {ans:.2f}%', (xh, xo), textcoords="offset points", xytext=(0, 20),
        #              ha='center', fontsize=20)
    # print(x_val)
    # print(y_val)
    #  = X[x_val[0], y_val[0]]]
    # [Y[x_val[0], y_val[0]]
    if VLC == 0:
        for i in range(len(x_val)):
            ax.scatter(X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]], marker='\u002E', c='black', s=1)
    else:
        for i in range(len(x_val)):
            ax.scatter(X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]], marker='\u002E', c='black', s=50)

    # ax.plot([X[x_val[0], y_val[0]] , X[x_val[-1],y_val[-1]], Y[x_val[-1], y_val[-1]],Y[x_val[-1], y_val[-1]]]
    #         , color='black', linestyle='-')
    # ax.plot([0, 0.2], [0.6, 0.6], color='black', linestyle='-')
    # ax.plot([0.2, 0.2], [0, 0.6], color='black', linestyle='-')
    # ax.plot([xh1, xh], [xo1, xo], color='black', linestyle='-')
    # ax.plot([xh1, xh], [xo1, xo], color='black', linestyle='-')
    # ax.plot([xh1, xh], [xo1, xo], color='black', linestyle='-')
    if VLC == 0:
        plt.annotate(f'$\eta^{{lim}}_{{STH}}={max1:.2f}\%$', (0.2, 0.1),
                     textcoords="offset points", xytext=(60, 0), ha='center', fontsize=15, c='black')
    else:
        plt.annotate(f'$\eta^{{lim}}_{{STH}}={max1:.2f}\%$', (X[x_val[-1], y_val[-1]], Y[x_val[-1], y_val[-1]]),
                     textcoords="offset points", xytext=(30, 20), ha='center', fontsize=15, c='black')
    if xh < 2 and xo < 2:
        for i in range(len1):
            xh_p = xh - (0.059 * i)
            xo_p = xo + (0.059 * i)
            xh_p = round(xh_p, 2)
            xo_p = round(xo_p, 2)
            if i == 0:
                plt.annotate(f'0', (xh_p, xo_p), textcoords="offset points", xytext=(30, 10),
                             ha='center', fontsize=15)
                plt.annotate(f'pH', (xh_p, xo_p), textcoords="offset points", xytext=(30, -10),
                             ha='center', fontsize=15)
            else:
                plt.annotate(f'{i}', (xh_p, xo_p), textcoords="offset points", xytext=(30, 10),
                             ha='center', fontsize=15)
            ax.scatter(xh_p, xo_p, marker='*', c='black')
    # if xh1 < 2 and xo1 < 2:
    #     plt.annotate(f'pH={t1}', (xh1, xo1), textcoords="offset points", xytext=(30, 0),
    #                  ha='center', fontsize=15)
    #     ax.scatter(xh1, xo1, marker='*', c='black')

    ax.plot([xh1, xh], [xo1, xo], color='black', linestyle='-')

    # #添加colorbar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)  # 对colorbar的大小进行设置

    cbar.ax.invert_yaxis()

    cbar.update_ticks()  # 显示colorbar的刻度值

    font3 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 33,
             }
    ax.set_xlabel('$\chi$(H$_2$) (eV)', font3)
    ax.set_ylabel('$\chi$(O$_2$) (eV)', font3)
    # 设置主刻度的标签， 带入主刻度旋转角度和字体大小参数
    ax.set_xticks([0, 0.4, 0.8, 1.2, 1.6, 2])
    ax.set_xticklabels(['0', '0.4', '0.8', '1.2', '1.6', '2'], fontsize=30, weight='bold', family='Times New Roman')
    ax.set_yticks([0, 0.4, 0.8, 1.2, 1.6, 2])
    ax.set_yticklabels(['', '0.4', '0.8', '1.2', '1.6', '2'], fontsize=30, weight='bold', family='Times New Roman')

    cbar.ax.tick_params(width=3, length=8)
    ax.set_xticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
    ax.set_yticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
    ax.tick_params(direction='in', which='both')
    ax.tick_params(axis='x', which='major', width=3, length=8)
    ax.tick_params(axis='x', which='minor', width=2, length=4)
    ax.tick_params(axis='y', which='major', width=3, length=8)
    ax.tick_params(axis='y', which='minor', width=2, length=4)
    # plt.rcParams['figure.figsize']=(6,4)
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    # plt.gcf().subplots_adjust(left=0.05, top=0.91, bottom=0.09)
    plt.tight_layout()
    file_path = os.path.join(s, "Xh_Xo_sth_Z.png")
    fig.savefig(file_path)
    #
    plt.show()
    mpl.rcParams.update(mpl.rcParamsDefault)
    # fig.savefig(r'D:\tu\python\myplot_0_g.png')
    return Xh,Xo,Z
def coSTH_Z_pu(VLC,xh,xo,s,t1):
    if not os.path.exists(s):
        os.makedirs(s)
    workbook1 = xlrd.open_workbook(r'D:\down\1.xls')  # 文件路径读文件
    names1 = workbook1.sheet_names()
    xh = round(xh,2)
    xo = round(xo,2)
    objsth = workbook1.sheet_by_name("Sheet1")
    Xh = np.linspace(0, 2, 201)
    Xo = np.linspace(0, 2, 201)

    xh1 = xh - (t1 * 0.059)
    xo1 = xo + (t1 * 0.059)

    xh1 = round(xh1, 2)
    xo1 = round(xo1, 2)
    # VLC = 0

    X, Y = np.meshgrid(Xh, Xo)

    Eg = X + Y + 1.23 - VLC

    l = objsth.col_values(2)
    f = objsth.col_values(3)
    h = objsth.col_values(4)
    n = objsth.col_values(5)
    # 第2行开始
    l.pop(0)
    f.pop(0)
    h.pop(0)
    n.pop(0)
    Emax = np.max(l)
    emax = np.max(h)
    # 求最小梯度
    DEmin = np.min(np.gradient(l))
    dEmin = np.min(np.gradient(h))
    r = []
    Z = []
    max1 = 0
    min1 = 1000
    ans = 0
    ans1 = 0
    for i, Eg0 in zip(Xo, Eg):

        for j, Eg1 in zip(Xh, Eg0):
            # print(i,j,Eg1)
            if Eg1 < 0:
                r.append(np.nan)
                continue
            if j >= 0.2 and i >= 0.6:
                if Eg1 < 0.31:
                    Eg1 = 0.31
                E = Eg1
            elif j < 0.2 and i >= 0.6:
                E = Eg1 + 0.2 - j
                if E < 0.31:
                    E = 0.31
                if Eg1 < 0.31:
                    Eg1 = 0.31
            elif j >= 0.2 and i < 0.6:
                E = Eg1 + 0.6 - i
                if E < 0.31:
                    E = 0.31
                if Eg1 < 0.31:
                    Eg1 = 0.31
            elif j < 0.2 and i < 0.6:
                E = Eg1 + 0.8 - i - j
                if E < 0.31:
                    E = 0.31
                if Eg1 < 0.31:
                    Eg1 = 0.31
            # 插值积分
            Eintp = np.arange(Eg1, DEmin + Emax, DEmin)  # 无除
            eintp = np.arange(E, emax + dEmin, dEmin)  # 有除
            Jintp = np.interp(Eintp, l, f)
            jintp = np.interp(eintp, h, n)
            fintp = np.interp(Eintp, h, n)
            Egg = np.trapz(Jintp, Eintp)
            Eh = np.trapz(jintp, eintp)
            Ef = np.trapz(fintp, Eintp)
            nabs = Egg / 1000.37/2
            ncu = Eh * 1.23 / Egg
            STH = nabs * ncu
            correctedSTH = STH * 1000.37 / (1000.37 + VLC * Ef)
            if round(xh,2) == round(j,2) and round(xo,2) == round(i,2):
                ans = correctedSTH * 100
            if round(xh1,2) == round(j,2) and round(xo1,2) == round(i,2):
                ans1 = correctedSTH * 100
            if correctedSTH >= max1:
                max1 = correctedSTH
            if correctedSTH < min1:
                min1 = correctedSTH

            r.append(correctedSTH * 100)
        Z.append(r)
        r = []

    # print(max)
    # print(min)
    # for i in Z:
    #     print(i)
    # a = 10
    # b = 8   "#86190d",,"#0c08ed"
    max1 = max1 * 100
    min1 = min1 * 100
    fig, ax = plt.subplots()
    # plt.figure(figsize=(12, 8))
    fig.set_size_inches(10, 8)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    colors = ["#86190d", "#dc0105", "#ffad0d", "#edfc1b", "#8efd72", "#1ffee1", "#03bffe", "#0c08ed"]
    # colors = ["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#da0300","#7e0004"]
    colors.reverse()
    cmap = LinearSegmentedColormap.from_list("custom_rainbow", colors)
    cmap_reversed = cmap.reversed()
    norm = mpl.colors.Normalize(vmin=min1, vmax=max1)
    # ax.rc('font', size=12, family='serif')
    mpl.rc('font', size=33, family='Times New Roman', weight='bold')
    # 刻度线值, manual=manual_locations
    # manual_locations = [(0.23, 0.64), (0.3, 0.72), (0.42, 0.8), (0.5, 0.83), (0.8, 0.85)]
    C = ax.contour(X, Y, Z, colors='black', linewidths=0)

    cs = ax.contourf(X, Y, Z, 5, cmap=cmap)  # 画出等高图

    x_val, y_val = np.where(np.isclose(Z, max1))

    # 在图中标记Z值为0.5的点
    # plt.scatter(xh, xo, color='blue')  # 用红色标出你要标记的点
    # title = ["pH", '\u03c7h', '\u03c7o', "\u03B7abs", "\u03B7cu", "\u03B7STH", '\u03B7\u2032STH']


    # print(x_val)
    # print(y_val)
    #  = X[x_val[0], y_val[0]]]
    # [Y[x_val[0], y_val[0]]


    # #添加colorbar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)  # 对colorbar的大小进行设置

    cbar.ax.invert_yaxis()

    cbar.update_ticks()  # 显示colorbar的刻度值

    font3 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 33,
             }
    ax.set_xlabel('$\chi$(H$_2$) (eV)', font3)
    ax.set_ylabel('$\chi$(O$_2$) (eV)', font3)
    # 设置主刻度的标签， 带入主刻度旋转角度和字体大小参数
    ax.set_xticks([0, 0.4, 0.8, 1.2, 1.6, 2])
    ax.set_xticklabels(['0', '0.4', '0.8', '1.2', '1.6', '2'], fontsize=30, weight='bold', family='Times New Roman')
    ax.set_yticks([0, 0.4, 0.8, 1.2, 1.6, 2])
    ax.set_yticklabels(['', '0.4', '0.8', '1.2', '1.6', '2'], fontsize=30, weight='bold', family='Times New Roman')

    cbar.ax.tick_params(width=3, length=8)
    ax.set_xticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
    ax.set_yticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
    ax.tick_params(direction='in', which='both')
    ax.tick_params(axis='x', which='major', width=3, length=8)
    ax.tick_params(axis='x', which='minor', width=2, length=4)
    ax.tick_params(axis='y', which='major', width=3, length=8)
    ax.tick_params(axis='y', which='minor', width=2, length=4)
    # plt.rcParams['figure.figsize']=(6,4)
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    # plt.gcf().subplots_adjust(left=0.05, top=0.91, bottom=0.09)
    plt.tight_layout()
    file_path = os.path.join(s, "Xh_Xo_sth_Z_pu.png")
    fig.savefig(file_path)
    #
    # plt.show()
    mpl.rcParams.update(mpl.rcParamsDefault)
    # fig.savefig(r'D:\tu\python\myplot_0_g.png')
    return Xh,Xo,Z

def CBM_VBM_Z(VLC,C,V,s,t1,EA,EB):
    if not os.path.exists(s):
        os.makedirs(s)
    workbook1 = xlrd.open_workbook(r'D:\down\1.xls')  # 文件路径读文件
    names1 = workbook1.sheet_names()
    # print(names1)
    C = round(C,2)
    V = round(V,2)
    objsth = workbook1.sheet_by_name("Sheet1")
    # Xh = np.linspace(0, 1, 101)
    # Xo = np.linspace(0, 1, 101)
    # VLC = 0

    # Y, X = np.meshgrid(Xh, Xo)
    #
    # Eg = X + Y +1.23 - VLC

    l = objsth.col_values(2)
    f = objsth.col_values(3)
    h = objsth.col_values(4)
    n = objsth.col_values(5)
    # 第2行开始
    l.pop(0)
    f.pop(0)
    h.pop(0)
    n.pop(0)


    emax = np.max(h)
    # 求最小梯度
    DEmin = np.min(np.gradient(l))
    dEmin = np.min(np.gradient(h))
    r = []
    Z = []
    ans = 0
    max1 = 0
    min1 = 1000
    CBM = np.arange(-4.44, -3.23, 0.01)
    VBM = np.arange(-7.3, -5.66, 0.01)
    X, Y = np.meshgrid(CBM, VBM)
    Eg = X - Y
    for i, Eg0 in zip(VBM, Eg):
        for j, Eg1 in zip(CBM, Eg0):
            # print(i,j,Eg1)
            xh = j + 4.44 + VLC - (t1 * 0.059)
            xo = -5.67 - i + - (t1 * 0.059)
            if xh>=0 and xo>=0:
                E = max(EA,EB)
                if E<0.31:
                    E = 0.31

            Eintp = np.arange(Eg1, DEmin + emax, DEmin)  # 无除
            eintp = np.arange(E, emax + dEmin, dEmin)  # 有除
            Jintp = np.interp(Eintp, l, f)
            jintp = np.interp(eintp, h, n)
            fintp = np.interp(Eintp, h, n)
            Egg = np.trapz(Jintp, Eintp)
            Eh = np.trapz(jintp, eintp)
            Ef = np.trapz(fintp, Eintp)
            nabs = Egg / 1000.37/2
            ncu = Eh * 1.23 / Egg
            # STH = Eh * 1.23 / 1000.37/2
            STH = nabs*ncu
            correctedSTH = STH*1000.37/(1000.37+VLC*Ef)
            if C == round(j,2) and V == round(i,2):
                ans = correctedSTH * 100
            if correctedSTH >= max1:
                max1 = correctedSTH
            if correctedSTH < min1:
                min1 = correctedSTH

            r.append(correctedSTH * 100)
        Z.append(r)
        r = []

    max1 = max1 * 100
    min1 = min1 * 100
    # print(max)
    # print(min)
    fig, ax = plt.subplots(figsize=(10, 8))
    # plt.figure(figsize=(10, 8))
    fig.set_size_inches(10, 8)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    colors = ["#86190d", "#dc0105", "#ffad0d", "#edfc1b", "#8efd72", "#1ffee1", "#03bffe", "#0c08ed"]
    # colors = ["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#da0300","#7e0004"]
    colors.reverse()
    cmap = LinearSegmentedColormap.from_list("custom_rainbow", colors)
    cmap_reversed = cmap.reversed()
    norm = mpl.colors.Normalize(vmin=min1, vmax=max1)
    # ax.rc('font', size=12, family='serif')
    mpl.rc('font', size=33, family='Times New Roman', weight='bold')
    cs = ax.contourf(X, Y, Z, cmap=cmap)  # 画出等高图

    x_val, y_val = np.where(np.isclose(Z, max1))
    # print(x_val)
    # print(y_val)

    # 在图中标记Z值为0.5的点\u2032
    # title = ["pH", '\u03c7h', '\u03c7o', "\u03B7abs", "\u03B7cu", "\u03B7STH", '\u03B7\u2032STH']
    # plt.annotate(f'({C:.2f}, {V:.2f})', (C, V), textcoords="offset points", xytext=(0, -20),#坐标
    #              ha='center', fontsize=20)
    if VLC == 0:
        plt.annotate(f'{ans:.2f}%', (C, V), textcoords="offset points", xytext=(30, 10),
                     ha='center', fontsize=15)

    else:
        plt.annotate(f'{ans:.2f}%', (C, V), textcoords="offset points", xytext=(30, 10),
                     ha='center', fontsize=15)
    if VLC == 0:
        for i in range(len(x_val)):
            ax.scatter(X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]], marker='*', c='black', s=1)
        plt.annotate(f'$\eta^{{lim}}_{{STH}}={max1:.2f}\%$', (-4.3, -6.1),
                     textcoords="offset points", xytext=(60, 30), ha='center', fontsize=15, c='black')
    else:
        for i in range(len(x_val)):
            ax.scatter(X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]], marker='*', c='black', s=50)
        plt.annotate(f'$\eta^{{lim}}_{{STH}}={max1:.2f}\%$', (X[x_val[-1], y_val[-1]], Y[x_val[-1], y_val[-1]]),
                     textcoords="offset points", xytext=(60, 20), ha='center', fontsize=15, c='black')
        # plt.annotate('*', (X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]]), fontsize=20)

    if C >= -4.44 and C < -2.23 and V >= -7.3 and V < -4.66:
        ax.scatter(C, V, marker='*', c='black')
        plt.annotate(f'pH={t1}', (C, V), textcoords="offset points", xytext=(30, -10),
                     ha='center', fontsize=15)
    #
    # ax.colorbars(label='correctSTH')
    # #添加colorbar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)  # 对colorbar的大小进行设置
    # ticks = [1, 3, 6, 9, 12, 15]  # 刻度值，包括最大值
    # tick_labels = ['1', '3', '6', '9', '12', '15']  # 对应的刻度标签
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels(tick_labels)
    #
    # # 使用FixedLocator和FixedFormatter
    # cbar.locator = FixedLocator(ticks)
    # cbar.formatter = FixedFormatter(tick_labels)
    cbar.ax.invert_yaxis()
    # cbar.set_ticks([10,15,20,25,30,35,max])
    # cbar.set_ticklabels(['10', '15', '20', '25', '30','35','38.18'])
    # cbar.ax.set_title('η’$_{STH}$ $\%$',fontsize=40,weight='bold', family='Times New Roman',pad=15)
    cbar.update_ticks()  # 显示colorbar的刻度值
    # ticks = cbar.get_ticks()  # 获取当前的刻度标签
    # new_ticks = ticks +0  # 向上移动2个单位
    # cbar.set_ticks(new_ticks)  # 设置新的刻度标签

    font3 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 33,
             }
    ax.set_xlabel('CBM (eV)', font3)
    ax.set_ylabel('VBM (eV)', font3)
    # 设置主刻度的标签， 带入主刻度旋转角度和字体大小参数
    ax.set_xticks([-4.44, -4.24, -4.04, -3.84, -3.64, -3.44, -3.23])
    ax.set_xticklabels(['-4.44', '-4.24', '-4.04', '-3.84', '-3.64', '-3.43', '-3.24'], fontsize=24, weight='bold',
                       family='Times New Roman')
    ax.set_yticks([-7, -6.7, -6.4, -6.1, -5.9, -5.67])
    ax.set_yticklabels(['-7', '-6.7', '-6.4', '-6.1', '-5.9', '-5.67'], fontsize=24, weight='bold',
                       family='Times New Roman')

    cbar.ax.tick_params(width=3, length=8)
    # ax.set_xticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
    # ax.set_yticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
    ax.tick_params(direction='in', which='both')
    ax.tick_params(axis='x', which='major', width=3, length=8)
    # ax.tick_params(axis='x', which='minor', width=2, length=4)
    ax.tick_params(axis='y', which='major', width=3, length=8)
    # ax.tick_params(axis='y', which='minor', width=2, length=4)
    # plt.rcParams['figure.figsize']=(6,4)
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    # plt.gcf().subplots_adjust(left=0.05, top=0.91, bottom=0.09)

    plt.tight_layout()
    file_path = os.path.join(s, "CBM_VBM_sth_Z.png")
    fig.savefig(file_path)
    #
    plt.show()
    mpl.rcParams.update(mpl.rcParamsDefault)
    # fig.savefig(r'D:\tu\python\CBM_VBM_0.png')
    return CBM,VBM,Z
def CBM_VBM_Z_J(VLC,C,V,s,t1):
    if not os.path.exists(s):
        os.makedirs(s)
    workbook1 = xlrd.open_workbook(r'D:\down\1.xls')  # 文件路径读文件
    names1 = workbook1.sheet_names()
    # print(names1)
    C = round(C,2)
    V = round(V,2)
    objsth = workbook1.sheet_by_name("Sheet1")
    # Xh = np.linspace(0, 1, 101)
    # Xo = np.linspace(0, 1, 101)
    # VLC = 0

    # Y, X = np.meshgrid(Xh, Xo)
    #
    # Eg = X + Y +1.23 - VLC

    l = objsth.col_values(2)
    f = objsth.col_values(3)
    h = objsth.col_values(4)
    n = objsth.col_values(5)
    # 第2行开始
    l.pop(0)
    f.pop(0)
    h.pop(0)
    n.pop(0)


    emax = np.max(h)
    # 求最小梯度
    DEmin = np.min(np.gradient(l))
    dEmin = np.min(np.gradient(h))
    r = []
    Z = []
    ans = 0
    max = 0
    min = 1000
    CBM = np.arange(-6.44, -3.23, 0.01)
    VBM = np.arange(-7.3, -5.66, 0.01)
    X, Y = np.meshgrid(CBM, VBM)
    Eg = X - Y
    for i, Eg0 in zip(VBM, Eg):
        for j, Eg1 in zip(CBM, Eg0):
            # print(i,j,Eg1)
            xh = j + 4.44 + VLC - (t1 * 0.059)
            xo = -5.67 - i + - (t1 * 0.059)
            if Eg1 < 0:
                r.append(np.nan)
                continue
            if xh >= 0.2 and xo >= 0.6:
                if Eg1 < 0.31:
                    Eg1 = 0.31
                E = Eg1
            elif xh < 0.2 and xo >= 0.6:
                E = Eg1 + 0.2 - xh
                if E < 0.31:
                    E = 0.31
                if Eg1 < 0.31:
                    Eg1 = 0.31
            elif xh >= 0.2 and xo < 0.6:
                E = Eg1 + 0.6 - xo
                if E < 0.31:
                    E = 0.31
                if Eg1 < 0.31:
                    Eg1 = 0.31
            elif xh < 0.2 and xo < 0.6:
                E = Eg1 + 0.8 - xh - xo
                if E < 0.31:
                    E = 0.31
                if Eg1 < 0.31:
                    Eg1 = 0.31

            Eintp = np.arange(Eg1, DEmin + emax, DEmin)  # 无除
            eintp = np.arange(E, emax + dEmin, dEmin)  # 有除
            Jintp = np.interp(Eintp, l, f)
            jintp = np.interp(eintp, h, n)
            fintp = np.interp(Eintp, h, n)
            Egg = np.trapz(Jintp, Eintp)
            Eh = np.trapz(jintp, eintp)
            Ef = np.trapz(fintp, Eintp)
            nabs = Egg / 1000.37/2
            ncu = Eh * 1.23 / Egg
            # STH = Eh * 1.23 / 1000.37/2
            STH = nabs*ncu
            correctedSTH = STH*1000.37/(1000.37+VLC*Ef)
            if C == round(j,2) and V == round(i,2):
                ans = correctedSTH * 100
            if correctedSTH >= max:
                max = correctedSTH
            if correctedSTH < min:
                min = correctedSTH

            r.append(correctedSTH * 100)
        Z.append(r)
        r = []

    max = max * 100
    min = min * 100
    print(max)
    print(min)
    fig, ax = plt.subplots(figsize=(10, 8))
    # plt.figure(figsize=(10, 8))
    fig.set_size_inches(10, 8)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    colors = ["#86190d", "#dc0105", "#ffad0d", "#edfc1b", "#8efd72", "#1ffee1", "#03bffe", "#0c08ed"]
    # colors = ["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#da0300","#7e0004"]
    colors.reverse()
    cmap = LinearSegmentedColormap.from_list("custom_rainbow", colors)
    cmap_reversed = cmap.reversed()
    norm = mpl.colors.Normalize(vmin=min, vmax=max)
    # ax.rc('font', size=12, family='serif')
    mpl.rc('font', size=33, family='Times New Roman', weight='bold')
    cs = ax.contourf(X, Y, Z, cmap=cmap)  # 画出等高图

    x_val, y_val = np.where(np.isclose(Z, max))
    # print(x_val)
    # print(y_val)

    # 在图中标记Z值为0.5的点\u2032
    # title = ["pH", '\u03c7h', '\u03c7o', "\u03B7abs", "\u03B7cu", "\u03B7STH", '\u03B7\u2032STH']
    # plt.annotate(f'({C:.2f}, {V:.2f})', (C, V), textcoords="offset points", xytext=(0, -20),#坐标
    #              ha='center', fontsize=20)
    if VLC == 0:
        plt.annotate(f'{ans:.2f}%', (C, V), textcoords="offset points", xytext=(30, 10),
                     ha='center', fontsize=15)

    else:
        plt.annotate(f'{ans:.2f}%', (C, V), textcoords="offset points", xytext=(30, 10),
                     ha='center', fontsize=15)
    if VLC == 0:
        for i in range(len(x_val)):
            ax.scatter(X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]], marker='*', c='black', s=1)
        plt.annotate(f'$\eta^{{lim}}_{{STH}}={max:.2f}\%$', (-4.3, -6.4),
                     textcoords="offset points", xytext=(0, -10), ha='center', fontsize=15, c='black')
    else:
        for i in range(len(x_val)):
            ax.scatter(X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]], marker='*', c='black', s=50)
        plt.annotate(f'$\eta^{{lim}}_{{STH}}={max:.2f}\%$', (X[x_val[-1], y_val[-1]], Y[x_val[-1], y_val[-1]]),
                     textcoords="offset points", xytext=(60, -20), ha='center', fontsize=15, c='black')
        # plt.annotate('*', (X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]]), fontsize=20)

    if C >= -6.44 and C < -2.23 and V >= -7.3 and V < -4.66:
        ax.scatter(C, V, marker='*', c='black')
        plt.annotate(f'pH={t1}', (C, V), textcoords="offset points", xytext=(30, -10),
                     ha='center', fontsize=15)
    #
    # ax.colorbars(label='correctSTH')
    # #添加colorbar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)  # 对colorbar的大小进行设置
    # ticks = [1, 3, 6, 9, 12, 15]  # 刻度值，包括最大值
    # tick_labels = ['1', '3', '6', '9', '12', '15']  # 对应的刻度标签
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels(tick_labels)
    #
    # # 使用FixedLocator和FixedFormatter
    # cbar.locator = FixedLocator(ticks)
    # cbar.formatter = FixedFormatter(tick_labels)
    cbar.ax.invert_yaxis()
    # cbar.set_ticks([10,15,20,25,30,35,max])
    # cbar.set_ticklabels(['10', '15', '20', '25', '30','35','38.18'])
    # cbar.ax.set_title('η’$_{STH}$ $\%$',fontsize=40,weight='bold', family='Times New Roman',pad=15)
    cbar.update_ticks()  # 显示colorbar的刻度值
    # ticks = cbar.get_ticks()  # 获取当前的刻度标签
    # new_ticks = ticks +0  # 向上移动2个单位
    # cbar.set_ticks(new_ticks)  # 设置新的刻度标签

    font3 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 33,
             }
    ax.set_xlabel('CBM (eV)', font3)
    ax.set_ylabel('VBM (eV)', font3)
    # 设置主刻度的标签， 带入主刻度旋转角度和字体大小参数
    ax.set_xticks([ -5, -4.5, -4, -3.5])
    ax.set_xticklabels([ '-5', '-4.5', '-4', '-3.5'], fontsize=24, weight='bold',
                       family='Times New Roman')
    ax.set_yticks([-7, -6.6, -6.2, -5.8])
    ax.set_yticklabels(['-7', '-6.6', '-6.2', '-5.8'], fontsize=24, weight='bold',
                       family='Times New Roman')

    cbar.ax.tick_params(width=3, length=8)
    # ax.set_xticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
    # ax.set_yticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
    ax.tick_params(direction='in', which='both')
    ax.tick_params(axis='x', which='major', width=3, length=8)
    # ax.tick_params(axis='x', which='minor', width=2, length=4)
    ax.tick_params(axis='y', which='major', width=3, length=8)
    # ax.tick_params(axis='y', which='minor', width=2, length=4)
    # plt.rcParams['figure.figsize']=(6,4)
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    # plt.gcf().subplots_adjust(left=0.05, top=0.91, bottom=0.09)

    plt.tight_layout()
    file_path = os.path.join(s, "CBM_VBM_sth_Z.png")
    fig.savefig(file_path)
    #
    plt.show()
    mpl.rcParams.update(mpl.rcParamsDefault)
    # fig.savefig(r'D:\tu\python\CBM_VBM_0.png')
    return CBM,VBM,Z
def CBM_VBM_Z_pu(VLC,C,V,s):
    if not os.path.exists(s):
        os.makedirs(s)
    workbook1 = xlrd.open_workbook(r'D:\down\1.xls')  # 文件路径读文件
    names1 = workbook1.sheet_names()
    # print(names1)
    C = round(C,2)
    V = round(V,2)
    objsth = workbook1.sheet_by_name("Sheet1")
    # Xh = np.linspace(0, 1, 101)
    # Xo = np.linspace(0, 1, 101)
    # VLC = 0

    # Y, X = np.meshgrid(Xh, Xo)
    #
    # Eg = X + Y +1.23 - VLC

    l = objsth.col_values(2)
    f = objsth.col_values(3)
    h = objsth.col_values(4)
    n = objsth.col_values(5)
    # 第2行开始
    l.pop(0)
    f.pop(0)
    h.pop(0)
    n.pop(0)


    emax = np.max(h)
    # 求最小梯度
    DEmin = np.min(np.gradient(l))
    dEmin = np.min(np.gradient(h))
    r = []
    Z = []
    ans = 0
    max = 0
    min = 1000
    CBM = np.arange(-4.44, -3.23, 0.01)
    VBM = np.arange(-7.3, -5.66, 0.01)
    X, Y = np.meshgrid(CBM, VBM)
    Eg = X - Y
    for i, Eg0 in zip(VBM, Eg):
        for j, Eg1 in zip(CBM, Eg0):
            # print(i,j,Eg1)
            xh = j + 4.44 + VLC
            xo = -5.67 - i
            if (xh < 0 or xo < 0) or Eg1 < 1.23:
                r.append(np.nan)
                continue
            if xh >= 0.2 and xo >= 0.6:
                E = Eg1

            elif xh < 0.2 and xo >= 0.6:
                E = Eg1 + 0.2 - xh

            elif xh >= 0.2 and xo < 0.6:
                E = Eg1 + 0.6 - xo

            elif xh < 0.2 and xo < 0.6:
                E = Eg1 + 0.8 - xh - xo

            Eintp = np.arange(Eg1, DEmin + emax, DEmin)  # 无除
            eintp = np.arange(E, emax + dEmin, dEmin)  # 有除
            Jintp = np.interp(Eintp, l, f)
            jintp = np.interp(eintp, h, n)
            fintp = np.interp(Eintp, h, n)
            Egg = np.trapz(Jintp, Eintp)
            Eh = np.trapz(jintp, eintp)
            Ef = np.trapz(fintp, Eintp)
            nabs = Egg / 1000.37/2
            ncu = Eh * 1.23 / Egg
            # STH = Eh * 1.23 / 1000.37/2
            STH = nabs*ncu
            correctedSTH = STH*1000.37/(1000.37+VLC*Ef)
            if C == round(j) and V == round(i):
                ans = correctedSTH * 100
            if correctedSTH >= max:
                max = correctedSTH
            if correctedSTH < min:
                min = correctedSTH

            r.append(correctedSTH * 100)
        Z.append(r)
        r = []

    # for i in Z:
    #     print(i)
    # a = 10
    # b = 8   "#86190d",,"#0c08ed"
    max = max * 100
    min = min * 100
    # print(max)
    # print(min)
    fig, ax = plt.subplots()
    # plt.figure(figsize=(12, 8))
    fig.set_size_inches(10, 8)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    colors = ["#86190d", "#dc0105", "#ffad0d", "#edfc1b", "#8efd72", "#1ffee1", "#03bffe", "#0c08ed"]
    # colors = ["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#da0300","#7e0004"]
    colors.reverse()
    cmap = LinearSegmentedColormap.from_list("custom_rainbow", colors)
    cmap_reversed = cmap.reversed()
    norm = mpl.colors.Normalize(vmin=min, vmax=max)
    # ax.rc('font', size=12, family='serif')
    mpl.rc('font', size=33, family='Times New Roman', weight='bold')
    cs = ax.contourf(X, Y, Z, cmap=cmap)  # 画出等高图

    x_val, y_val = np.where(np.isclose(Z, max))
    # print(x_val)
    # print(y_val)

    # 在图中标记Z值为0.5的点
    # title = ["pH", '\u03c7h', '\u03c7o', "\u03B7abs", "\u03B7cu", "\u03B7STH", '\u03B7\u2032STH']

    # ax.colorbars(label='correctSTH')
    # #添加colorbar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)  # 对colorbar的大小进行设置
    # ticks = [1, 3, 6, 9, 12, 15]  # 刻度值，包括最大值
    # tick_labels = ['1', '3', '6', '9', '12', '15']  # 对应的刻度标签
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels(tick_labels)
    #
    # # 使用FixedLocator和FixedFormatter
    # cbar.locator = FixedLocator(ticks)
    # cbar.formatter = FixedFormatter(tick_labels)
    cbar.ax.invert_yaxis()
    # cbar.set_ticks([10,15,20,25,30,35,max])
    # cbar.set_ticklabels(['10', '15', '20', '25', '30','35','38.18'])
    # cbar.ax.set_title('η’$_{STH}$ $\%$',fontsize=40,weight='bold', family='Times New Roman',pad=15)
    cbar.update_ticks()  # 显示colorbar的刻度值
    # ticks = cbar.get_ticks()  # 获取当前的刻度标签
    # new_ticks = ticks +0  # 向上移动2个单位
    # cbar.set_ticks(new_ticks)  # 设置新的刻度标签

    font3 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 33,
             }
    ax.set_xlabel('CBM (eV)', font3)
    ax.set_ylabel('VBM (eV)', font3)
    # 设置主刻度的标签， 带入主刻度旋转角度和字体大小参数
    ax.set_xticks([-4.44, -4.24, -4.04, -3.84, -3.64, -3.44, -3.23])
    ax.set_xticklabels(['-4.44', '-4.24', '-4.04', '-3.84', '-3.64', '-3.43', '-3.23'], fontsize=30, weight='bold',
                       family='Times New Roman')
    ax.set_yticks([ -7, -6.7, -6.4, -6.1, -5.9, -5.68])
    ax.set_yticklabels([ '-7', '-6.7', '-6.4', '-6.1', '-5.9', '-5.68'], fontsize=30, weight='bold',
                       family='Times New Roman')

    cbar.ax.tick_params(width=3, length=8)
    # ax.set_xticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
    # ax.set_yticks([0.2, 0.4, 1, 1.4, 1.8], minor=True)
    ax.tick_params(direction='in', which='both')
    ax.tick_params(axis='x', which='major', width=3, length=8)
    # ax.tick_params(axis='x', which='minor', width=2, length=4)
    ax.tick_params(axis='y', which='major', width=3, length=8)
    # ax.tick_params(axis='y', which='minor', width=2, length=4)
    # plt.rcParams['figure.figsize']=(6,4)
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    # plt.gcf().subplots_adjust(left=0.05, top=0.91, bottom=0.09)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.tight_layout()
    file_path = os.path.join(s, "CBM_VBM_sth_Z_pu.png")
    fig.savefig(file_path)
    #
    # plt.show()
    mpl.rcParams.update(mpl.rcParamsDefault)
    # fig.savefig(r'D:\tu\python\CBM_VBM_0.png')
    return CBM,VBM,Z
def Delta_Eg(s):
    if not os.path.exists(s):
        os.makedirs(s)
    workbook1 = xlrd.open_workbook(r'D:\down\1.xls')  # 文件路径读文件
    names1 = workbook1.sheet_names()
    objsth = workbook1.sheet_by_name("Sheet1")

    l = objsth.col_values(2)
    f = objsth.col_values(3)
    h = objsth.col_values(4)
    n = objsth.col_values(5)
    # 第2行开始
    l.pop(0)
    f.pop(0)
    h.pop(0)
    n.pop(0)
    # Emin = np.min(l)
    Emax = np.max(l)
    # emin = min(h)
    emax = np.max(h)
    # 求最小梯度
    DEmin = np.min(np.gradient(l))
    dEmin = np.min(np.gradient(h))
    Vlc = np.arange(0, 5.1, 0.01)
    Eg = np.arange(0, 3.1, 0.01)
    X, Y = np.meshgrid(Vlc, Eg)
    r = []
    Z = []
    max = 0
    max1 = 0
    min = 1000
    flag = 1
    for Eg1 in Eg:
        for i in Vlc:
            flag = 1
            for xh in np.arange(0, 1, 0.1):
                xo = Eg1 - 1.23 + i - xh
                correctedSTH = 0
                if xo < 0 and flag == 1:
                    correctedSTH = np.nan
                    max = np.nan
                    break
                elif xo < 0:
                    break
                else:
                    if xh >= 0.2 and xo >= 0.6:
                        E = Eg1
                    elif xh < 0.2 and xo >= 0.6:
                        E = Eg1 + 0.2 - xh
                    elif xh >= 0.2 and xo < 0.6:
                        E = Eg1 + 0.6 - xo
                    elif xh < 0.2 and xo < 0.6:
                        E = Eg1 + 0.8 - xh - xo
                    # 插值积分
                    Eintp = np.arange(Eg1, DEmin + Emax, DEmin)
                    eintp = np.arange(E, emax + dEmin, dEmin)
                    Jintp = np.interp(Eintp, l, f)
                    jintp = np.interp(eintp, h, n)
                    fintp = np.interp(Eintp, h, n)
                    Egg = np.trapz(Jintp, Eintp)
                    Eh = np.trapz(jintp, eintp)
                    Ef = np.trapz(fintp, Eintp)
                    nabs = Egg / 1000.37
                    ncu = Eh * 1.23 / Egg
                    STH = nabs * ncu
                    correctedSTH = STH * 1000.37 / (1000.37 + i * Ef)
                    flag = 0
                if correctedSTH * 100 > max:
                    max = correctedSTH * 100
            if max1 < max:
                max1 = max
            if max < min:
                min = max



            r.append(max)
            max = 0
        Z.append(r)
        r = []

    max = max1
    min = min
    fig, ax = plt.subplots()
    # plt.figure(figsize=(12, 8))
    fig.set_size_inches(14, 8)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    colors = ["#86190d", "#dc0105", "#ffad0d", "#edfc1b", "#8efd72", "#1ffee1", "#03bffe", "#0c08ed"]
    # colors = ["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#da0300","#7e0004"]
    colors.reverse()
    cmap = LinearSegmentedColormap.from_list("custom_rainbow", colors)
    cmap_reversed = cmap.reversed()
    norm = mpl.colors.Normalize(vmin=min, vmax=max)
    # ax.rc('font', size=12, family='serif')
    mpl.rc('font', size=30, family='Times New Roman', weight='bold')
    # 在图中text（）,(0.68,0.35),(0.72,0.45),(0.61,0.21),
    # lev = [40,36,32,28,24,20,16,12]
    # lev.reverse()
    # manual_locations = [(1.7,0.8),(2,1.2),(2.2,1.4),(2.6,1.8),(3,2.2),(3.5,2.6)](1.5,0.6),
    manual_locations = [(1.7, 0.8), (1.95, 1.1), (2.15, 1.2), (2.4, 1.7), (2.6, 1.8), (2.8, 2.2), (2.8, 2.7)]
    C = ax.contour(X, Y, Z, colors='black', linewidths=0)
    labels = ax.clabel(C, inline=True, fontsize=35, manual=manual_locations)
    for label in labels:
        # label.set_position((label.get_position()[0], label.get_position()[1])-0.03)
        x, y = label.get_position()
        label.set_position((x, y - 0.01))
    #
    cs = ax.contourf(X, Y, Z, 7, cmap=cmap)  # 画出等高图
    #
    x_val, y_val = np.where(np.isclose(Z, max))


    # 在等高图上画出这些点，使用星形符号

    # 在图中标记Z值为0.5的点
    for i in range(len(x_val)):
        # plt.annotate('*', (X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]]), fontsize=20)
        ax.scatter(X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]], marker='*', c='white', s=300)
        # plt.annotate(f'$\eta^{{max}}_{{STH}}={max:.2f}\%$', (X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]]), textcoords="offset points", xytext=(0, -20),
        #          ha='center', fontsize=20)

    # ax.colorbars(label='correctSTH')
    # #添加colorbar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)  # 对colorbar的大小进行设置
    ticks = [5, 10, 15, 20, 25, 30, 35, 40]  # 刻度值，包括最大值
    tick_labels = ['5', '10', '15', '20', '25', '30', '35', '40']  # 对应的刻度标签

    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.invert_yaxis()
    # 使用FixedLocator和FixedFormatter
    cbar.locator = FixedLocator(ticks)
    cbar.formatter = FixedFormatter(tick_labels)

    # cbar.set_ticks([10,15,20,25,30,35,max])
    # cbar.set_ticklabels(['10', '15', '20', '25', '30','35','38.18'])
    # cbar.ax.set_title('η’$_{STH}$ $\%$',fontsize=40,weight='bold', family='Times New Roman',pad=15)
    cbar.update_ticks()  # 显示colorbar的刻度值
    # ticks = cbar.get_ticks()  # 获取当前的刻度标签
    # new_ticks = ticks +0  # 向上移动2个单位
    # cbar.set_ticks(new_ticks)  # 设置新的刻度标签

    font3 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 40,
             }
    ax.set_xlabel('\u0394\u03A6 (eV)', font3)
    ax.set_ylabel('Band gap (eV)', font3)
    # 设置主刻度的标签， 带入主刻度旋转角度和字体大小参数
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(['0', '1', '2', '3', '4', '5'], fontsize=35, weight='bold', family='Times New Roman')
    ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.set_yticklabels(['', '0.5', '1', '1.5', '2', '2.5', '3'], fontsize=35, weight='bold', family='Times New Roman')

    ax.tick_params(direction='in', which='both')
    ax.tick_params(axis='x', which='major', width=3, length=8)
    ax.tick_params(axis='y', which='major', width=3, length=8)
    # plt.rcParams['figure.figsize']=(6,4)
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    # plt.gcf().subplots_adjust(left=0.05, top=0.91, bottom=0.09)
    plt.tight_layout()
    file_path = os.path.join(s, "VLC_Eg_sth.png")
    fig.savefig(file_path)
    #
    plt.show()
    mpl.rcParams.update(mpl.rcParamsDefault)
    # fig.savefig(r'D:\tu\python\myplot_Eg.png')


    return Vlc,Eg,Z

def Delta_Eg_STH_Z(s):
    if not os.path.exists(s):
        os.makedirs(s)
    workbook1 = xlrd.open_workbook(r'D:\down\1.xls')  # 文件路径读文件
    # names1 = workbook1.sheet_names()
    # print(names1)
    objsth = workbook1.sheet_by_name("Sheet1")

    l = objsth.col_values(2)
    f = objsth.col_values(3)
    h = objsth.col_values(4)
    n = objsth.col_values(5)
    # 第2行开始
    l.pop(0)
    f.pop(0)
    h.pop(0)
    n.pop(0)

    Emax = np.max(l)

    emax = np.max(h)
    # 求最小梯度
    DEmin = np.min(np.gradient(l))
    dEmin = np.min(np.gradient(h))
    Vlc = np.arange(0, 5.1, 0.1)
    Eg = np.arange(0, 3.1, 0.1)
    max1 = 0
    X, Y = np.meshgrid(Vlc, Eg)
    r = []
    Z = []
    max = 0
    max1 = 0
    min = 1000
    flag = 1
    for Eg1 in Eg:
        for i in Vlc:
            flag = 1
            for xh in np.arange(0, 1, 0.1):
                xo = Eg1 - 1.23 + i - xh
                correctedSTH = 0
                if xo < 0 and flag == 1:
                    correctedSTH = np.nan
                    max = np.nan
                    break
                elif xo < 0:
                    break
                else:
                    if xh >= 0.2 and xo >= 0.6:
                        E = Eg1
                    elif xh < 0.2 and xo >= 0.6:
                        E = Eg1 + 0.2 - xh
                    elif xh >= 0.2 and xo < 0.6:
                        E = Eg1 + 0.6 - xo
                    elif xh < 0.2 and xo < 0.6:
                        E = Eg1 + 0.8 - xh - xo
                    # 插值积分
                    Eintp = np.arange(Eg1, DEmin + Emax, DEmin)
                    eintp = np.arange(E, emax + dEmin, dEmin)
                    Jintp = np.interp(Eintp, l, f)
                    jintp = np.interp(eintp, h, n)
                    fintp = np.interp(Eintp, h, n)
                    Egg = np.trapz(Jintp, Eintp)
                    Eh = np.trapz(jintp, eintp)
                    Ef = np.trapz(fintp, Eintp)
                    nabs = Egg / 1000.37
                    ncu = Eh * 1.23 / Egg
                    STH = nabs * ncu
                    correctedSTH = STH * 1000.37 / (1000.37 + i * Ef)/2
                    flag = 0
                if correctedSTH * 100 > max:
                    max = correctedSTH * 100
            if max1 < max:
                max1 = max
            if max < min:
                min = max

            # ?

            r.append(max)
            max = 0
        Z.append(r)
        r = []
    # for i in Z:
    #     print(i)
    # print(max1)
    # print(min)
    max = max1
    min = min
    fig, ax = plt.subplots()
    # plt.figure(figsize=(12, 8))
    fig.set_size_inches(14, 8)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    colors = ["#86190d", "#dc0105", "#ffad0d", "#edfc1b", "#8efd72", "#1ffee1", "#03bffe", "#0c08ed"]
    # colors = ["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#da0300","#7e0004"]
    colors.reverse()
    cmap = LinearSegmentedColormap.from_list("custom_rainbow", colors)
    cmap_reversed = cmap.reversed()
    norm = mpl.colors.Normalize(vmin=min, vmax=max)
    # ax.rc('font', size=12, family='serif')
    mpl.rc('font', size=30, family='Times New Roman', weight='bold')
    # 在图中text（）,(0.68,0.35),(0.72,0.45),(0.61,0.21),
    # lev = [40,36,32,28,24,20,16,12]
    # lev.reverse(), manual=manual_locations
    # manual_locations = [(1.7,0.8),(2,1.2),(2.2,1.4),(2.6,1.8),(3,2.2),(3.5,2.6)](1.5,0.6),
    manual_locations = [(1.7, 0.8), (1.95, 1.1), (2.15, 1.2), (2.4, 1.7), (2.6, 1.8), (2.8, 2.2), (2.8, 2.7)]
    C = ax.contour(X, Y, Z, colors='black', linewidths=0)
    labels = ax.clabel(C, inline=True, fontsize=35,manual=manual_locations)
    for label in labels:
        # label.set_position((label.get_position()[0], label.get_position()[1])-0.03)
        x, y = label.get_position()
        label.set_position((x, y - 0.01))
    # for label in labels:
    #     label.set_position((label.get_position()[0], label.get_position()[1]))
    # labels(1).set_position((labels(1).get_position()[0]-0.03, labels(1).get_position()[1])-0.03)
    # labels[3].set_position((labels[3].get_position()[0], labels[3].get_position()[1])+0.03)
    # x, y = labels[3].get_position()
    # labels[3].set_position((x + 0.03, y+0.03))
    # x, y = labels[1].get_position()
    # labels[1].set_position((x - 0.03, y-0.03))

    # plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    #
    cs = ax.contourf(X, Y, Z, 7, cmap=cmap)  # 画出等高图

    x_val, y_val = np.where(np.isclose(Z, max))

    # 在等高图上画出这些点，使用星形符号

    # 在图中标记Z值为0.5的点
    for i in range(len(x_val)):
        # plt.annotate('*', (X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]]), fontsize=20)
        ax.scatter(X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]], marker='*', c='black', s=20)
        plt.annotate(f'$\eta^{{lim}}_{{STH}}={max:.2f}\%$', (X[x_val[i], y_val[i]], Y[x_val[i], y_val[i]]),
                     textcoords="offset points", xytext=(0, -20),
                     ha='center', fontsize=20)
    #
    # ax.colorbars(label='correctSTH')
    # #添加colorbar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)  # 对colorbar的大小进行设置
    # ticks = [5, 10, 15, 20, 25, 30, 35, 40]  # 刻度值，包括最大值
    # tick_labels = ['5', '10', '15', '20', '25', '30', '35', '40']  # 对应的刻度标签

    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels(tick_labels)

    # # 使用FixedLocator和FixedFormatter
    # cbar.locator = FixedLocator(ticks)
    # cbar.formatter = FixedFormatter(tick_labels)
    cbar.ax.invert_yaxis()
    # cbar.set_ticks([10,15,20,25,30,35,max])
    # cbar.set_ticklabels(['10', '15', '20', '25', '30','35','38.18'])
    # cbar.ax.set_title('η’$_{STH}$ $\%$',fontsize=40,weight='bold', family='Times New Roman',pad=15)
    cbar.update_ticks()  # 显示colorbar的刻度值
    # ticks = cbar.get_ticks()  # 获取当前的刻度标签
    # new_ticks = ticks +0  # 向上移动2个单位
    # cbar.set_ticks(new_ticks)  # 设置新的刻度标签



    font3 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 40,
             }
    ax.set_xlabel('$\Delta$$\Phi$ (eV)', font3)
    ax.set_ylabel('Band gap (eV)', font3)
    # 设置主刻度的标签， 带入主刻度旋转角度和字体大小参数
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(['0', '1', '2', '3', '4', '5'], fontsize=35, weight='bold', family='Times New Roman')
    ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.set_yticklabels(['', '0.5', '1', '1.5', '2', '2.5', '3'], fontsize=35, weight='bold', family='Times New Roman')

    ax.tick_params(direction='in', which='both')
    # plt.rcParams['figure.figsize']=(6,4)
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    # plt.gcf().subplots_adjust(left=0.05, top=0.91, bottom=0.09)
    plt.tight_layout()
    file_path = os.path.join(s, "VLC_Eg_sth_Z.png")
    fig.savefig(file_path)
    #
    plt.show()
    mpl.rcParams.update(mpl.rcParamsDefault)
    # fig.savefig(r'D:\tu\python\myplot_Eg.png')
    return Vlc,Eg,Z
def PCE():
    # 单位换算 E=h/波长
    nm2eV = 1.2398419739e-6 * 1e9
    # 读取文件
    # dat = csvread('C:\Users\17130\Documents\WeChat Files\wxid_g9zl93hbm4so22\FileStorage\File\2023-11\ASTMG1731.csv',2);
    readfile1 = xlrd.open_workbook(r"D:\daima\PCE\AS.xls")
    # l=dat(:,1); f=dat(:,3);
    obj = readfile1.sheet_by_name("Sheet1")
    # print(obj1)
    # print(obj1.nrows)
    # print(obj1.ncols)
    #
    val0 = obj.col_values(0)
    val2 = obj.col_values(2)

    val0.pop(0)
    val0.pop(0)
    val2.pop(0)
    val2.pop(0)
    # 用上面的公式换算出能量和能量分布，并按照递增排序，方便进行积分运算和插值
    x = np.array(val0)
    y = np.array(val2)

    # E=nm2eV./flipud(l);
    def flipud(l):
        return l[::-1]

    E = nm2eV / flipud(x)
    print(type(E[0]))
    # J=flipud(f.*l)./E;
    J = flipud(x * y) / E
    # % 取能量最大值最小值（确定积分上下限），求梯度（数据最小间隔），并用梯形法计算积分总功率
    Emin = min(E)
    Emax = max(E)

    # Emin=min(E); Emax=max(E);

    # dEmin=min(gradient(E));
    dEmin = min(np.gradient(E))

    # Jtot=trapz(E, J)
    Jtot = np.trapz(J, E)

    # %%计算单个效率值
    Eopt = 1.000642
    Ec = 0.536142
    # Eintp=[Eopt:dEmin:Emax];
    Eintp = np.arange(Eopt, Emax, dEmin)
    # Jintp=interp1(E,J, Eintp, 'linear');
    # Jsc=trapz(Eintp, Jintp./Eintp)/Jtot;
    Jintp = np.interp(Eintp, E, J)
    Jsc = np.trapz(Jintp / Eintp, Eintp) / Jtot
    # print(Jsc)
    cs = 0.65 * (Eopt - Ec - 0.3) * Jsc * 100
    print(cs)
    # %% 作图（效率和Eopt、Ec的二维图）
    # Eopt=[1.5:.01:3.1];
    # Ec=[0:.01:1.0];
    Eopt = np.arange(0.5, 3.01, 0.01)
    Ec = np.arange(0, 1.01, 0.01)
    x, y = np.meshgrid(Eopt, Ec)

    # Jsc=zeros(1,length(Eopt));
    # Jsc = np.zeros(len(Eopt))
    # print(Jsc)
    Js = []
    Jsc = []
    eta = []
    eta1 = []
    # for i=1:length(Eopt)-1
    # Eintp=[Eopt(i):dEmin:Emax];
    # Jintp=interp1(E,J, Eintp, 'linear');
    # Jsc(i)=trapz(Eintp, Jintp./Eintp)/Jtot;
    # end
    max = 0
    min = 10000
    for i in range(len(Ec)):
        for j in range(len(Eopt)):
            Eintp = np.arange(Eopt[j], Emax, dEmin)
            Jintp = np.interp(Eintp, E, J)
            Js.append(np.trapz(Jintp / Eintp, Eintp) / Jtot)
        Jsc.append(Js)
        Js = []
    for x1, y1, Js in zip(x, y, Jsc):
        for x2, y2, js in zip(x1, y1, Js):
            ans = 0.65 * (x2 - y2 - 0.3) * js * 100
            if (x2 - y2 - 0.3) < 0:
                eta1.append(np.nan)
            else:
                if max < ans:
                    max = ans
                if min > ans:
                    min = ans
                eta1.append(ans)
        eta.append(eta1)
        eta1 = []
    # eta = 0.65*(x-y-0.3)*Jsc*100
    # [x, y]=meshgrid(Eopt, Ec);

    # %生成网格矩阵
    # [z, y]=meshgrid(Jsc, Ec);
    # z, y = np.meshgrid(Jsc, Ec)
    # print(Jsc[0])
    # for i in Jsc:
    #     for j in i:
    #         print(j)
    # print((x-y-0.3)*0.65*100*Jsc)

    # for i in eta:
    #     for j in i:
    #         print(j)
    print(max)
    print(min)
    # %填充的二维等高线图
    # contourf(x,y,eta, [2:1:22], 'ShowText','on')
    # %坐标轴修改
    # set(gca,'FontSize',20);
    # xlabel('Optical gap (eV)','fontsize',24)
    # ylabel('Conduction band offset (eV)','fontsize',24)
    fig, ax = plt.subplots()
    # plt.figure(figsize=(12, 8))
    fig.set_size_inches(10, 8)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    colors = ["#86190d", "#dc0105", "#ffad0d", "#edfc1b", "#8efd72", "#1ffee1", "#0c08ed", "#03bffe"]
    # colors = ["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#da0300","#7e0004"]
    colors.reverse()
    cmap = LinearSegmentedColormap.from_list("custom_rainbow", colors)
    norm = mpl.colors.Normalize(vmin=min, vmax=max)
    mpl.rc('font', size=33, family='Times New Roman', weight='bold')
    plt.contourf(x, y, eta, 22, cmap=cmap)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)  # 对colorbar的大小进行设置
    ticks = [2, 5, 8, 11, 14, 17, 20]  # 刻度值，包括最大值
    tick_labels = ['2', '5', '8', '11', '14', '17', '20']  # 对应的刻度标签
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.invert_yaxis()
    # 使用FixedLocator和FixedFormatter
    cbar.locator = FixedLocator(ticks)
    cbar.formatter = FixedFormatter(tick_labels)
    # C = ax.contour(x, y, z,colors='black', linewidths=0)
    # labels = ax.clabel(C, inline=True, fontsize=35)
    # plt.colorbar(label='eta')
    x_val, y_val = np.where(np.isclose(eta, max))
    print(x_val)
    print(y_val)
    # plt.annotate('*',xy=(Eopt, Ec), xytext=(Eopt, Ec),fontsize=6)
    # 在图中标记Z值为0.5的点
    for i in range(len(x_val)):
        plt.annotate('*', (x[x_val[i], y_val[i]], y[x_val[i], y_val[i]]), fontsize=6)

    font3 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 33,
             }
    ax.set_xlabel('E$_g$ (eV)', font3)
    ax.set_ylabel('$\Delta$E$_c$(eV)', font3)
    ax.set_xticks([0.5, 1, 1.5, 2, 2.5, 3])
    ax.set_xticklabels(['0.5', '1.0', '1.5', '2.0', '2.5', '3.0'], fontsize=30, weight='bold', family='Times New Roman')
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(['', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=30, weight='bold', family='Times New Roman')

    ax.tick_params(direction='in', which='both')

    plt.tight_layout()

    plt.show()
    mpl.rcParams.update(mpl.rcParamsDefault)
    # fig.savefig(r'D:\tu\python\PCE_cs.png')

    pass

def Zhexian_Delta_Eg_STH(s):
    if not os.path.exists(s):
        os.makedirs(s)
    nm2eV = 1.2398419739e-6 * 1e9
    workbook = xlwt.Workbook()
    sheet0 = workbook.add_sheet('1')
    workbook1 = xlrd.open_workbook(r'D:\down\1.xls')  # 文件路径读文件
    objsth = workbook1.sheet_by_name("Sheet1")
    h0 = objsth.col_values(0)
    h1 = objsth.col_values(1)
    l = objsth.col_values(2)
    f = objsth.col_values(3)
    h = objsth.col_values(4)
    n = objsth.col_values(5)

    # 第2行开始
    h0.pop(0)
    h1.pop(0)
    l.pop(0)
    f.pop(0)
    h.pop(0)
    n.pop(0)

    # print(type(h[1]))
    def flipud(l):
        return l[::-1]

    h0_r = flipud(h0)
    h1_r = flipud(h1)
    # print(h1)
    # print((h0[0]*h1[1])/(nm2eV/h0[0]))
    E = [nm2eV / x for x in h0]
    J = [(x * y) / z for x, y, z in zip(h0, h1, E)]
    Jh = [j / e for j, e in zip(J, E)]
    emax = np.max(E)
    emin = np.min(E)
    dEmin = min(np.gradient(E))
    DEmin = min(np.gradient(l))
    # dEmin = min(np.gradient(h))
    STH = []
    Eg1 = []
    x = []
    max = 0
    pp = 0
    for i in np.arange(0, 10, 0.01):
        E = 2.03 - i
        if i < 1.72:
            Eg = 2.03 - i
        else:
            Eg = 0.31
        if E < 0.31:
            E = 0.31
        Eintp = np.arange(Eg, emax + dEmin, dEmin)
        eintp = np.arange(E, emax + dEmin, dEmin)
        fintp = np.interp(Eintp, h, n)
        jintp = np.interp(eintp, h, n)
        Eh = np.trapz(jintp, eintp)
        Ef = np.trapz(fintp, eintp)
        y = 1.23 * Eh / 1000.37
        yl = y * 1000.37 / (1000.37 + i * Ef) * 100
        # print(yl)
        # print(E)
        if yl>max:
            max = yl
            pp = i
        STH.append(yl)
        Eg1.append(Eg)
        x.append(i)
    fig, ax1 = plt.subplots()
    fig.set_size_inches(11, 7)

    ax2 = ax1.twinx()

    ax1.plot(x, STH, color='b',linewidth=2)
    ax2.plot(x, Eg1,  color='g',linewidth=2)

    x2 = [1.72, 5, 5, 1.72]
    y2 = [0, 0, 0.31, 0.31]

    ax2.fill(x2, y2, 'g')  # 'g'
    # # 画一条水平的虚线
    # ax1.axhline(y=max, color='black', linestyle='--', xmin=0, xmax=pp)
    #
    # # 画一条垂直的虚线
    # ax1.axvline(x=pp, color='black', linestyle='--', ymin=0, ymax=max)

    ax1.plot([0, pp], [max, max], color='black', linestyle='--')
    ax1.plot([pp, pp], [0, max], color='black', linestyle='--')

    ax2.plot([0, 1.72], [0.31, 0.31], color='black', linestyle='--')
    ax2.plot([1.72, 1.72], [0, 0.31], color='black', linestyle='--')

    ax1.annotate(f'{max:.2f}%', (pp, max), textcoords="offset points", xytext=(10, 10),
                 ha='center', fontsize=20, fontweight='bold')
    # ax1.annotate(f'{0.31:.2f}', (0.5, 0.31), textcoords="offset points", xytext=(0, 10),
    #              ha='center', fontsize=10)

    # ax2.annotate(f'(1.72,0.31)', (1.72, 0.31), textcoords="offset points", xytext=(25, 20),
    #              ha='center', fontsize=20,color = 'g', fontweight='bold')
    ax2.scatter(1.72, 0.31, marker='*', c='black', s=300,zorder=3)
    # ax1.annotate(f'{1.72:.2f}', (1.72, 0), textcoords="offset points", xytext=(0, -20),
    #              ha='center', fontsize=20,weight='bold')

    ax1.set_ylim(0, 50)
    ax2.set_ylim(0, 6)
    ax1.set_xlim(0, 5)

    ax1.set_xticks([0, 1, 2, 3, 4, 5])
    ax1.set_xticklabels(['0', '1', '2', '3', '4', '5'], fontsize=30, weight='bold', family='Times New Roman')
    ax1.set_yticks([10, 20, 30, 40])
    ax1.set_yticklabels(['10', '20', '30', '40'], fontsize=30, weight='bold', family='Times New Roman',color='b')
    ax2.set_yticks([0, 2, 4, 6])
    ax2.set_yticklabels(['0', '2', '4', '6'], fontsize=30, weight='bold', family='Times New Roman',color='g')

    ax1.tick_params(direction='in', which='both')
    ax2.tick_params(direction='in', which='both')
    font3 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 33,
             }
    # 设置横纵坐标范围
    ax1.set_xlabel('\u0394\u03A6 (eV)', font3)
    ax1.set_ylabel('Efficiency (%)', color='b', fontsize=30, weight='bold', family='Times New Roman')
    ax2.set_ylabel('Band gap (eV)', color='g', fontsize=30, weight='bold', family='Times New Roman')
    plt.title('')

    ax2.spines['right'].set_color('green')
    ax2.spines['right'].set_linewidth(3)
    ax1.spines['left'].set_color('blue')
    ax1.spines['left'].set_linewidth(3)
    ax1.spines['top'].set_linewidth(3)
    ax1.spines['bottom'].set_linewidth(3)
    # 显示图例
    # ax1.legend(loc='upper left',fontsize=20)
    # ax2.legend(loc='upper right',fontsize=20)
    plt.tight_layout()

    file_path = os.path.join(s, "Dotted_VLC_Eg_sth.png")
    fig.savefig(file_path)

    plt.show()
    mpl.rcParams.update(mpl.rcParamsDefault)

    return x,STH,Eg1

def Zhexian_Delta_Eg_STH_Z(s):
    if not os.path.exists(s):
        os.makedirs(s)
    nm2eV = 1.2398419739e-6 * 1e9
    workbook = xlwt.Workbook()
    sheet0 = workbook.add_sheet('1')
    workbook1 = xlrd.open_workbook(r'D:\down\1.xls')  # 文件路径读文件
    objsth = workbook1.sheet_by_name("Sheet1")
    h0 = objsth.col_values(0)
    h1 = objsth.col_values(1)
    l = objsth.col_values(2)
    f = objsth.col_values(3)
    h = objsth.col_values(4)
    n = objsth.col_values(5)

    # 第2行开始
    h0.pop(0)
    h1.pop(0)
    l.pop(0)
    f.pop(0)
    h.pop(0)
    n.pop(0)

    # print(type(h[1]))
    def flipud(l):
        return l[::-1]

    h0_r = flipud(h0)
    h1_r = flipud(h1)
    # print(h1)
    # print((h0[0]*h1[1])/(nm2eV/h0[0]))
    E = [nm2eV / x for x in h0]
    J = [(x * y) / z for x, y, z in zip(h0, h1, E)]
    Jh = [j / e for j, e in zip(J, E)]
    emax = np.max(E)
    emin = np.min(E)
    dEmin = min(np.gradient(E))
    DEmin = min(np.gradient(l))
    # dEmin = min(np.gradient(h))
    STH = []
    Eg1 = []
    x = []
    max = 0
    pp = 0
    for i in np.arange(0, 10, 0.01):
        E = 2.03 - i
        if i < 1.72:
            Eg = 2.03 - i
        else:
            Eg = 0.31
        if E < 0.31:
            E = 0.31
        Eintp = np.arange(Eg, emax + dEmin, dEmin)
        eintp = np.arange(E, emax + dEmin, dEmin)
        fintp = np.interp(Eintp, h, n)
        jintp = np.interp(eintp, h, n)
        Eh = np.trapz(jintp, eintp)
        Ef = np.trapz(fintp, eintp)
        y = 1.23 * Eh / 1000.37
        yl = y * 1000.37 / (1000.37 + i * Ef) * 100/2
        # print(yl)
        # print(E)
        if yl > max:
            max = yl
            pp = i
        STH.append(yl)
        Eg1.append(Eg)
        x.append(i)
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 8)

    ax2 = ax1.twinx()


    x2 = [1.72, 5, 5, 1.72]
    y2 = [0, 0, 0.31, 0.31]

    ax2.fill(x2, y2, 'g')  # 'g'
    # # 画一条水平的虚线
    # ax1.axhline(y=max, color='black', linestyle='--', xmin=0, xmax=pp)
    #
    # # 画一条垂直的虚线
    # ax1.axvline(x=pp, color='black', linestyle='--', ymin=0, ymax=max)
    ax1.plot(x, STH, label='\u03B7\u2032STH', color='b')
    ax2.plot(x, Eg1, label='Eg', color='g')

    ax1.plot([0, pp], [max, max], color='black', linestyle='--')
    ax1.plot([pp, pp], [0, max], color='black', linestyle='--')

    ax2.plot([0, 1.72], [0.31, 0.31], color='black', linestyle='--')
    ax2.plot([1.72, 1.72], [0, 0.31], color='black', linestyle='--')

    ax1.annotate(f'{max:.2f}%', (pp, max), textcoords="offset points", xytext=(10, 10),
                 ha='center', fontsize=20)
    # ax1.annotate(f'{0.31:.2f}', (0.5, 0.31), textcoords="offset points", xytext=(0, 10),
    #              ha='center', fontsize=10)

    ax2.annotate(f'{0.31:.2f}', (5, 0.31), textcoords="offset points", xytext=(25, 0),
                 ha='center', fontsize=20, weight='bold')

    ax1.annotate(f'{1.72:.2f}', (1.72, 0), textcoords="offset points", xytext=(0, -20),
                 ha='center', fontsize=20, weight='bold')

    ax1.set_ylim(0, 50)
    ax2.set_ylim(0, 6)
    ax1.set_xlim(0, 5)

    ax1.set_xticks([0, 1, 2, 3, 4, 5])
    ax1.set_xticklabels(['0', '1', '2', '3', '4', '5'], fontsize=30, weight='bold', family='Times New Roman')
    ax1.set_yticks([10, 20, 30, 40])
    ax1.set_yticklabels(['10', '20', '30', '40'], fontsize=30, weight='bold', family='Times New Roman')
    ax2.set_yticks([0, 2, 4, 6])
    ax2.set_yticklabels(['0', '2', '4', '6'], fontsize=30, weight='bold', family='Times New Roman')

    ax1.tick_params(direction='in', which='both')
    ax2.tick_params(direction='in', which='both')
    font3 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 33,
             }
    # 设置横纵坐标范围
    ax1.set_xlabel('$\Delta$$\Phi$', font3)
    ax1.set_ylabel('Efficiency (%)', color='b', fontsize=30, weight='bold', family='Times New Roman')
    ax2.set_ylabel('Band gap (eV)', color='g', fontsize=30, weight='bold', family='Times New Roman')
    plt.title('')

    ax2.spines['right'].set_color('green')
    ax2.spines['right'].set_linewidth(3)
    ax1.spines['left'].set_color('blue')
    ax1.spines['left'].set_linewidth(3)
    ax1.spines['top'].set_linewidth(3)
    ax1.spines['bottom'].set_linewidth(3)
    # 显示图例
    ax1.legend(loc='upper left', fontsize=20)
    ax2.legend(loc='upper right', fontsize=20)

    file_path = os.path.join(s, "Dotted_VLC_Eg_sth.png")
    fig.savefig(file_path)

    plt.show()
    mpl.rcParams.update(mpl.rcParamsDefault)

    return x, STH, Eg1

def Table(table,s):
    if not os.path.exists(s):
        os.makedirs(s)
    df = pd.DataFrame(table[1:], columns=table[0])
    # 使用matplotlib绘制表格  ,'alpha':1
    fig, ax = plt.subplots(1, 1)
    ax.axis('tight')
    ax.axis('off')
    # cellDict = {'facecolor':(1, 1, 1, 1)}

    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', edges='open')
    # 获取表格的每一行
    cells = table.get_celld()

    # 遍历每一行
    # print(len(df.index))
    # print(len(df.columns))
    for i in range(len(df.index)):
        # 如果是第一行（标题行），则添加顶部和底部的线条
        if i == 0:
            for j in range(len(df.columns)):
                # cells[i, j].visible_edges = 'T'
                cells[i, j].visible_edges = 'TB'
                cells[i, j].set_edgecolor('black')
                cells[i, j].set_linewidth(1)
        if i == len(df.index) - 1:
            for j in range(len(df.columns)):
                # cells[i, j].visible_edges = 'T'
                cells[i + 1, j].visible_edges = 'B'
                cells[i + 1, j].set_edgecolor('black')
                cells[i + 1, j].set_linewidth(1)
        # 对于其他行，只添加底部的线条
    file_path = os.path.join(s, "Table.png")
    fig.savefig(file_path)
    plt.show()
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.close('all')

    pass

def General_PH(xh,xo,Eg):
    workbook1 = xlrd.open_workbook(r'D:\down\1.xls')  # 文件路径
    names2 = workbook1.sheet_names()
    # print(names2)
    objsth = workbook1.sheet_by_name("Sheet1")
    # 取excel第3，4，5，6列
    l = objsth.col_values(2)
    f = objsth.col_values(3)
    h = objsth.col_values(4)
    n = objsth.col_values(5)
    # 第2行开始
    l.pop(0)
    f.pop(0)
    h.pop(0)
    n.pop(0)
    PH = []
    STH1 = []
    ans = 0
    C = []
    title = ["pH", '\u03c7(H2) (eV)', '\u03c7(O2) (eV)', "\u03B7abs (%)", "\u03B7cu (%)", "\u03B7STH (%)"]
    C.append(title)
    E = 0
    for pH in range(0, 15):
        Xh9 = xh - (pH * 0.059)
        Xo9 = xo + (pH * 0.059)
        if Xh9 > 0 and Xo9 > 0:
            PH.append(pH)
            # 最值
            Emax = max(l)
            # emin = min(h)
            emax = max(h)
            # 求最小梯度
            DEmin = min(np.gradient(l))
            dEmin = min(np.gradient(h))
            if Xh9 >= 0.2 and Xo9 >= 0.6:
                E = Eg
                if E < 0.31:
                    E = 0.31
                if Eg < 0.31:
                    Eg = 0.31
            elif Xh9 < 0.2 and Xo9 >= 0.6:
                E = Eg + 0.2 - Xh9
                if E < 0.31:
                    E = 0.31
                if Eg < 0.31:
                    Eg = 0.31
            elif Xh9 >= 0.2 and Xo9 < 0.6:
                E = Eg + 0.6 - Xo9
                if E < 0.31:
                    E = 0.31
                if Eg < 0.31:
                    Eg = 0.31
            elif Xh9 < 0.2 and Xo9 < 0.6:
                E = Eg + 0.8 - Xh9 - Xo9
                if E < 0.31:
                    E = 0.31
                if Eg < 0.31:
                    Eg = 0.31
            # 插值积分
            Eg = round(Eg, 2)
            E = round(E, 2)
            Eintp = np.arange(Eg, DEmin + Emax, DEmin)
            eintp = np.arange(E, emax + dEmin, dEmin)
            Jintp = np.interp(Eintp, l, f)
            jintp = np.interp(eintp, h, n)
            Egg = np.trapz(Jintp, Eintp)
            Eh = np.trapz(jintp, eintp)
            nabs = Egg / 1000.37
            ncu = Eh * 1.23 / Egg
            STH = nabs * ncu
            STH1.append(STH)
            nabs2 = "%.2f" % (nabs * 100)
            ncu2 = "%.2f" % (ncu * 100)
            STH2 = "%.2f" % (STH * 100)
            Xh2 = "%.2f" % Xh9
            Xo2 = "%.2f" % Xo9
            B = [ pH, Xh2,  Xo2, nabs2, ncu2,  STH2]  # 要输出的值
            C.append(B)
    return C, PH, STH1
    pass

def Heterjunction_Z_PH(Xh,Xo,EA,EB,Eg):
    workbook1 = xlrd.open_workbook(r'D:\down\1.xls')  # 文件路径
    names2 = workbook1.sheet_names()
    # print(names2)
    objsth = workbook1.sheet_by_name("Sheet1")
    # 取excel第3，4，5，6列
    l = objsth.col_values(2)
    f = objsth.col_values(3)
    h = objsth.col_values(4)
    n = objsth.col_values(5)
    # 第2行开始
    l.pop(0)
    f.pop(0)
    h.pop(0)
    n.pop(0)
    # 最值

    Emax = max(l)
    emax = max(h)
    # 求最小梯度
    DEmin = min(np.gradient(l))
    dEmin = min(np.gradient(h))
    PH = []
    STH1 = []
    C = []
    title = ["pH", '\u03c7(H2) (eV)', '\u03c7(O2) (eV)', "\u03B7abs (%)", "\u03B7cu (%)", "\u03B7STH (%)"]
    C.append(title)
    for pH in range(0, 15):
        Xh9 = Xh - (pH * 0.059)
        Xo9 = Xo + (pH * 0.059)
        if Xh9 > 0 and Xo9 > 0:
            PH.append(pH)
            E = max(EA,EB)
            # 插值积分
            Eintp = np.arange(Eg, DEmin + Emax, DEmin)
            eintp = np.arange(E, dEmin + emax, dEmin)
            Jintp = np.interp(Eintp, l, f)
            jintp = np.interp(eintp, h, n)
            Egg = np.trapz(Jintp, Eintp)
            Eh = np.trapz(jintp, eintp)
            nabs = Egg / 1000.37
            ncu = Eh * 1.23 / Egg
            STH = nabs * ncu/2
            STH1.append(STH)
            nabs2 = "%.2f" % (nabs * 100)
            ncu2 = "%.2f" % (ncu * 100)
            STH2 =  "%.2f" % (STH * 100)
            Xh2 = "%.2f" % Xh9
            Xo2 = "%.2f" % Xo9
            B = [pH,  Xh2,  Xo2,  nabs2, ncu2,  STH2]  # 要输出的值
            C.append(B)
    return C, PH, STH1
    pass
def Janus_PH(Xh,Xo,Eg,VLC):
    workbook1 = xlrd.open_workbook(r'D:\down\1.xls')  # 文件路径
    names2 = workbook1.sheet_names()
    # print(names2)
    objsth = workbook1.sheet_by_name("Sheet1")
    # 取excel第3，4，5，6列
    l = objsth.col_values(2)
    f = objsth.col_values(3)
    h = objsth.col_values(4)
    n = objsth.col_values(5)
    # 第2行开始
    l.pop(0)
    f.pop(0)
    h.pop(0)
    n.pop(0)
    print('xh=',Xh)
    print('xo=',Xo)

    # 最值
    Emin = min(l)
    Emax = max(l)
    emin = min(h)
    emax = max(h)
    # 求最小梯度
    DEmin = min(np.gradient(l))
    dEmin = min(np.gradient(h))
    PH = []
    STH1 = []
    C = []
    title = ["pH", '\u03c7(H2) (eV)', '\u03c7(O2) (eV)', "\u03B7abs (%)", "\u03B7cu (%)", "\u03B7STH (%)",'\u03B7\u2032STH (%)']
    C.append(title)
    for pH in range(0, 15):
        Xh9 = Xh - (pH * 0.059)
        Xo9 = Xo + (pH * 0.059)
        if Xh9 > 0 and Xo9 > 0:
            PH.append(pH)
            if Xh9 >= 0.2 and Xo9 >= 0.6:
                E = Eg
                if E<0.31:
                    E = 0.31
            elif Xh9 < 0.2 and Xo9 >= 0.6:
                E = Eg + 0.2 - Xh9
                if E<0.31:
                    E = 0.31
            elif Xh9 >= 0.2 and Xo9 < 0.6:
                E = Eg + 0.6 - Xo9
                if E<0.31:
                    E = 0.31
            elif Xh9 < 0.2 and Xo9 < 0.6:
                E = Eg + 0.8 - Xh9 - Xo9
                if E<0.31:
                    E = 0.31
            # 插值积分
            if Eg < 0.31:
                Eg = 0.31
            Eintp = np.arange(Eg, DEmin + Emax, DEmin)
            eintp = np.arange(E, emax + dEmin, dEmin)
            Jintp = np.interp(Eintp, l, f)
            jintp = np.interp(eintp, h, n)
            fintp = np.interp(Eintp, h, n)
            Egg = np.trapz(Jintp, Eintp)
            Eh = np.trapz(jintp, eintp)
            Ef = np.trapz(fintp, Eintp)
            nabs = Egg / 1000.37
            ncu = Eh * 1.23 / Egg
            STH = nabs * ncu
            correctedSTH = STH * 1000.37 / (1000.37 + VLC * Ef)
            STH1.append(correctedSTH)
            nabsw =  "%.2f" % (nabs * 100)
            ncuw =  "%.2f" % (ncu * 100)
            STHw =  "%.2f" % (STH * 100)
            correctedSTHw =  "%.2f" % (correctedSTH * 100)
            Xhw = "%.2f" % Xh9
            Xow = "%.2f" % Xo9
            B = [pH,  Xhw,  Xow,  nabsw, ncuw, STHw, correctedSTHw]  # 要输出的值
            C.append(B)
    # print(C)
    # print(PH)
    # print(STH1)
    return C, PH, STH1
    pass


def Janus_Heterjunction_PH(Xh,Xo,Eg,VLC):
    workbook1 = xlrd.open_workbook(r'D:\down\1.xls')  # 文件路径
    names2 = workbook1.sheet_names()
    # print(names2)
    objsth = workbook1.sheet_by_name("Sheet1")
    # 取excel第3，4，5，6列
    l = objsth.col_values(2)
    f = objsth.col_values(3)
    h = objsth.col_values(4)
    n = objsth.col_values(5)
    # 第2行开始
    l.pop(0)
    f.pop(0)
    h.pop(0)
    n.pop(0)
    # 最值
    Emin = min(l)
    Emax = max(l)
    emin = min(h)
    emax = max(h)
    # 求最小梯度
    DEmin = min(np.gradient(l))
    dEmin = min(np.gradient(h))
    PH = []
    STH1 = []
    C = []
    title = ["pH", '\u03c7(H2) (eV)', '\u03c7(O2) (eV)', "\u03B7abs (%)", "\u03B7cu (%)", "\u03B7STH (%)",'\u03B7\u2032STH (%)']
    C.append(title)
    for pH in range(0, 15):
        Xh9 = Xh - (pH * 0.059)
        Xo9 = Xo + (pH * 0.059)
        if Xh9 > 0 and Xo9 > 0:
            PH.append(pH)
            if Xh9 >= 0.2 and Xo9 >= 0.6:
                E = Eg
            elif Xh9 < 0.2 and Xo9 >= 0.6:
                E = Eg + 0.2 - Xh9
            elif Xh9 >= 0.2 and Xo9 < 0.6:
                E = Eg + 0.6 - Xo9
            elif Xh9 < 0.2 and Xo9 < 0.6:
                E = Eg + 0.8 - Xh9 - Xo9
            # 插值积分
            Eintp = np.arange(Eg, DEmin + Emax, DEmin)
            eintp = np.arange(E, emax + dEmin, dEmin)
            Jintp = np.interp(Eintp, l, f)
            jintp = np.interp(eintp, h, n)
            fintp = np.interp(Eintp, h, n)
            Egg = np.trapz(Jintp, Eintp)
            Eh = np.trapz(jintp, eintp)
            Ef = np.trapz(fintp, Eintp)
            nabs = Egg / 1000.37
            ncu = Eh * 1.23 / Egg
            STH = nabs * ncu
            correctedSTH = STH * 1000.37 / (1000.37 + VLC * Ef)
            STH1.append(correctedSTH)
            nabsw =  "%.2f" % (nabs * 100)
            ncuw = "%.2f" % (ncu * 100)
            STHw =  "%.2f" % (STH * 100)
            correctedSTHw =  "%.2f" % (correctedSTH * 100)
            Xhw = "%.2f" % Xh9
            Xow = "%.2f" % Xo9
            B = [pH, Xhw, Xow, nabsw, ncuw, STHw, correctedSTHw]  # 要输出的值
            C.append(B)
    # print(C)
    # print(PH)
    # print(STH1)
    return C, PH, STH1
    pass

def Janus_Z_PH(Xh,Xo,Eg,VLC):
    workbook1 = xlrd.open_workbook(r'D:\down\1.xls')  # 文件路径
    names2 = workbook1.sheet_names()
    # print(names2)
    objsth = workbook1.sheet_by_name("Sheet1")
    # 取excel第3，4，5，6列
    l = objsth.col_values(2)
    f = objsth.col_values(3)
    h = objsth.col_values(4)
    n = objsth.col_values(5)
    # 第2行开始
    l.pop(0)
    f.pop(0)
    h.pop(0)
    n.pop(0)
    # 最值
    # Emin = min(l)
    Emax = np.max(l)
    # emin = min(h)
    emax = np.max(h)
    # 求最小梯度
    DEmin = np.min(np.gradient(l))
    dEmin = np.min(np.gradient(h))
    PH = []
    STH1 = []
    C = []
    title = ["pH", '\u03c7(H2) (eV)', '\u03c7(O2) (eV)', "\u03B7abs (%)", "\u03B7cu (%)", "\u03B7STH (%)",'\u03B7\u2032STH (%)']
    C.append(title)
    for pH in range(0, 15):
        Xh9 = Xh - (pH * 0.059)
        Xo9 = Xo + (pH * 0.059)
        if Xh9 > 0 and Xo9 > 0:
            PH.append(pH)
            if Xh9 >= 0.2 and Xo9 >= 0.6:
                E = Eg
            elif Xh9 < 0.2 and Xo9 >= 0.6:
                E = Eg + 0.2 - Xh9
            elif Xh9 >= 0.2 and Xo9 < 0.6:
                E = Eg + 0.6 - Xo9
            elif Xh9 < 0.2 and Xo9 < 0.6:
                E = Eg + 0.8 - Xh9 - Xo9
            # 插值积分
            Eintp = np.arange(Eg, DEmin + Emax, DEmin)
            eintp = np.arange(E, emax + dEmin, dEmin)
            Jintp = np.interp(Eintp, l, f)
            jintp = np.interp(eintp, h, n)
            fintp = np.interp(Eintp, h, n)
            Egg = np.trapz(Jintp, Eintp)
            Eh = np.trapz(jintp, eintp)
            Ef = np.trapz(fintp, Eintp)
            nabs = Egg / 1000.37
            ncu = Eh * 1.23 / Egg/2
            STH = nabs * ncu
            correctedSTH = STH * 1000.37 / (1000.37 + VLC * Ef)
            STH1.append(correctedSTH)
            nabsw =  "%.2f" % (nabs * 100)
            ncuw =  "%.2f" % (ncu * 100)
            STHw =  "%.2f" % (STH * 100)
            correctedSTHw ="%.2f" % (correctedSTH * 100)
            Xhw = "%.2f" % Xh9
            Xow = "%.2f" % Xo9
            B = [pH,  Xhw,  Xow,  nabsw, ncuw, STHw, correctedSTHw]  # 要输出的值
            C.append(B)
    return C, PH, STH1
def axis_ph(x, y,s,g):
    mpl.rcParams.update(mpl.rcParamsDefault)
    if not os.path.exists(s):
        os.makedirs(s)
    fig, ax = plt.subplots()
    # canvas_width, canvas_height = fig.get_size_inches()
    #
    # # 设置图片大小，确保不超过画布大小
    # desired_width = min(canvas_width, 12)  # 设置图片宽度不超过12英寸
    # desired_height = min(canvas_height, 5)  # 设置图片高度不超过5英寸
    # fig.set_size_inches(desired_width, desired_height)

    # 其他绘图操作
    fig.set_size_inches(12, 5)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    # plt.figure()
    # title = ["pH", '\u03c7h', '\u03c7o', "\u03B7abs", "\u03B7cu", "\u03B7STH", '\u03B7\u2032STH']
    if g == 0:
        plt.plot(x, y, 'bs-', alpha=1, linewidth=2, label='$\eta_{STH}$')  # 'bo-'表示蓝色实线，数据点实心原点标注
    elif g == 1:
        plt.plot(x, y, 'bs-', alpha=1, linewidth=2, label='$\eta^{{\prime}}_{STH}$')  # 'bo-'表示蓝色实线，数据点实心原点标注
    ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
    plt.xticks(fontsize=33, fontweight='bold')
    plt.yticks(fontsize=33, fontweight='bold')
    ax.legend(fontsize=20)  # 显示上面的label
    ax.set_xlabel('pH', fontsize=30, fontweight='bold')  # x_label

    ax.tick_params(direction='in', labelsize=20)
    if g == 0:
        ax.set_ylabel('$\eta_{STH}$', fontsize=30, fontweight='bold')  # y_label
    elif g == 1:
        ax.set_ylabel('$\eta^{{\prime}}_{STH}$', fontsize=30, fontweight='bold')  # y_label
    # ax.tick_params(direction='in', which='both')
    # plt.ylim(0,0.5)#仅设置y轴坐标范围
    file_path = os.path.join(s, "pH_STH.png")
    plt.tight_layout()
    plt.legend(fontsize=20)
    plt.savefig(file_path)
    plt.show()
    mpl.rcParams.update(mpl.rcParamsDefault)
    pass

def Save_coSTH(xh,xo,sth,s,g):


    # 两个一维列表和一个二维列表
    list1 = [1, 2, 3]
    list2 = ['a', 'b', 'c']
    list3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    # 将数据保存到dat文件
    if not os.path.exists(s):
        os.makedirs(s)
    # title = ["pH", '\u03c7h', '\u03c7o', "\u03B7abs", "\u03B7cu", "\u03B7STH", '\u03B7\u2032STH']
    # 将数据保存到子文件夹中的dat文件
    file_path = os.path.join(s, "Xh_Xo_sth_data.dat")
    with open(file_path, "w") as file:
        # "{:<5}".format
        if g == 0:
            file.write("{:<10}{:<10}{:<10}\n".format('\u03c7(h) (eV)','\u03c7(o) (eV)','\u03B7STH (%)'))
        elif g == 1:
            file.write("{:<10}{:<10}{:<10}\n".format('\u03c7(h) (eV)', '\u03c7(o) (eV)', '\u03B7\u2032STH (%)'))
        # file.write(f"Xh\t\t\t\tXo\t\t\t\tSTH%\n")
        for i,j,k in zip(xh,xo,sth):
            i = float(i)
            j = float(j)
            file.write("{:<10.2f}{:<10.2f}".format(i,j))
            for k1 in k:
                k1 = float(k1)
                file.write("{:<10.2f}".format(k1))
            file.write('\n')

    pass

def read_me_1(s):
    # 打开文件，如果不存在则创建
    # 指定文件夹路径

    # 拼接文件路径
    file_path = os.path.join(s, 'ex1.txt')

    # 打开文件，如果不存在则创建
    with open(file_path, 'w') as file:
        # 写入内容
        file.write('pH可分解水范围为0-7，极限STH为 = 17.2%\n')
        file.write('以pu.png为后缀的图为可编辑的图\n')
        file.write('Python 文件操作示例\n')

    pass
def read_me_2(s):
    # 打开文件，如果不存在则创建
    with open('example.txt', 'w') as file:
        # 写入内容
        file.write('\n')
        file.write('Hello, World!\n')
        file.write('Python 文件操作示例\n')

    pass
def read_me_3(s):
    # 打开文件，如果不存在则创建
    with open('example.txt', 'w') as file:
        # 写入内容
        file.write('\n')
        file.write('Hello, World!\n')
        file.write('Python 文件操作示例\n')

    pass
def read_me_4(s):
    # 打开文件，如果不存在则创建
    with open('example.txt', 'w') as file:
        # 写入内容
        file.write('\n')
        file.write('Hello, World!\n')
        file.write('Python 文件操作示例\n')

    pass
def read_me_5(s):
    # 打开文件，如果不存在则创建
    with open('example.txt', 'w') as file:
        # 写入内容
        file.write('\n')
        file.write('Hello, World!\n')
        file.write('Python 文件操作示例\n')

    pass
def read_me_6(s):
    # 打开文件，如果不存在则创建
    with open('example.txt', 'w') as file:
        # 写入内容
        file.write('\n')
        file.write('Hello, World!\n')
        file.write('Python 文件操作示例\n')

    pass

def Save_CBM_VBM(C,V,Z,s,g):
    if not os.path.exists(s):
        os.makedirs(s)

    # 将数据保存到子文件夹中的dat文件
    file_path = os.path.join(s, "CBM_VBM_STH_data.dat")
    with open(file_path, "w") as file:
        if g == 0:

            file.write("{:<10}{:<10}{:<10}\n".format('CBM','VBM','\u03B7STH (%)'))

        elif g == 1:
            file.write("{:<10}{:<10}{:<10}\n".format('CBM', 'VBM', '\u03B7\u2032STH (%)'))
        for i,j,k in zip(C,V,Z):
            file.write("{:<10.2f}{:<10.2f}".format(i, j))
            for k1 in k:
                file.write("{:<10.2f}".format(k1))
            file.write('\n')

    pass


def Save_Table(Z,s):
    if not os.path.exists(s):
        os.makedirs(s)

    # 将数据保存到子文件夹中的dat文件
    file_path = os.path.join(s, "Table_data.dat")
    # title = ["pH", '\u03c7h', '\u03c7o', "\u03B7abs", "\u03B7cu", "\u03B7STH", '\u03B7\u2032STH']
    with open(file_path, "w") as file:
        if len(Z[0]) == 6:
            # file.write(f"pH\t\t\tXh\t\t\t\tXo\t\t\t\tnabs\t\t\tncu\t\t\t\tSTH\n")
            file.write("{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}\n".format("pH", '\u03c7h (eV)', '\u03c7o (eV)', "\u03B7abs (%)", "\u03B7cu (%)", "\u03B7STH (%)"))
            n = 0
            for i in Z:
                if n == 0:
                    n +=1
                    continue
                for j in i:
                    file.write("{:<10}".format(j))
                file.write('\n')
        else:
            file.write("{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}\n".format("pH", '\u03c7h (eV)', '\u03c7o (eV)', "\u03B7abs (%)", "\u03B7cu (%)", "\u03B7STH (%)", '\u03B7\u2032STH (%)'))
            n = 0
            for i in Z:
                if n == 0:
                    n += 1
                    continue
                for j in i:
                    file.write("{:<10}".format(j))
                file.write('\n')




    pass

def Save_pH_STH(ph,sth,s,g):
    if not os.path.exists(s):
        os.makedirs(s)

    # 将数据保存到子文件夹中的dat文件
    file_path = os.path.join(s, "pH_sth_data.dat")
    with open(file_path, "w") as file:
        if g == 1:
            file.write("{:<10}{:<10}\n".format('pH','\u03B7\u2032STH (%)'))
        elif g == 0:
            file.write("{:<10}{:<10}\n".format('pH', '\u03B7STH (%)'))
        for i,j in zip(ph,sth):
            file.write("{:<10}{:<10.2f}\n".format(i,j*100))
    pass

def Save_Delta_Eg(VLC,Eg,STH,s):
    if not os.path.exists(s):
        os.makedirs(s)

    # 将数据保存到子文件夹中的dat文件
    file_path = os.path.join(s, "VLC_Eg_sth_data.dat")
    with open(file_path, "w") as file:
        # file.write(f"VLC\t\t\tEg\t\t\tSTH%\n")
        file.write("{:<10}{:<10}{:<10}\n".format('\u0394\u03C6', 'Eg','\u03B7\u2032STH (%)'))
        for i, j, k in zip(VLC, Eg, STH):
            file.write("{:<10.2f}{:<10.2f}".format(i, j))
            for k1 in k:
                file.write("{:<10.2f}".format(k1))
            file.write('\n')



def Save_Zhexian_Delta_Eg_STH(x,sth,Eg,s):
    if not os.path.exists(s):
        os.makedirs(s)

    # 将数据保存到子文件夹中的dat文件
    file_path = os.path.join(s, "Broken_VLC_Eg_sth_data.dat")
    with open(file_path, "w") as file:
        # file.write(f"Delta_phi\t\tSTH%\tEg\n")
        file.write("{:<10}{:<10}{:<10}\n".format('\u0394\u03C6', '\u03B7\u2032STH (%)', 'Eg'))
        for i,j,k in zip(x,sth,Eg):
            file.write("{:<10}{:<10.2f}{:<10.2f}\n".format(i, j, k))
    pass

def General():
    table1_1 = [
        ['Choose', 'Type'],
        ['1', '真空能级设为0时,输入VBM,CBM'],
        ['2', '费米能级设为0时,输入VBM,CBM,真空能级'],
        ['0', '退出'],
        ['9', '返回']
    ]
    table1_2 = [

        ['上述数据图片'],
        ['\u03c7h，\u03c7o--->STH图谱'],
        ['CBM,VBM--->STH图谱'],
        ['pH--->STH图谱'],
        ['上述数据'],
        ['\u03c7h，\u03c7o--->STH数据'],
        ['CBM,VBM--->STH数据'],
        ['pH--->STH数据'],

    ]
    table1_3 = [
        ['choose', 'type'],
        ['1', '当前PH值下的图谱'],
        ['2', '可选择PH值下的图谱']
    ]
    D = []
    E = []
    F = []
    print(tabulate(table1_1, headers='firstrow', tablefmt='simple'))
    ges = int(input("选择："))
    if ges == 1:
        VBM = float(input('Please enter the VBM:'))
        CBM = float(input('Please enter the CBM:'))
        Eg = abs(CBM - VBM)
        xh = CBM + 4.44
        xo = -5.67 - VBM

        if xh < 0 or xo < 0:
            print("PH = 0时，不能分解水！！")
        flag1 = 0
        for ph in range(15):
            xh1 = xh - (ph * 0.059)
            xo1 = xo + (ph * 0.059)
            if xh1 >= 0 and xo1 >= 0:
                flag1 = 1
                D,E,F = General_PH(xh , xo , Eg)
                break
        if flag1 == 0:
            print('并且全PH值下都不可以分解水！')
            return
        elif flag1 == 1:
            print('In the following pH range, photocatalytic materials can split water')
            if E[0] == E[-1]:
                print("pH：", E[0], '时可分解水！')
            else:
                print("pH:", E[0], "-", E[-1])
            print(tabulate(D, headers='firstrow', tablefmt='simple', floatfmt=".2f"))
            # t1 =input('选择绘制CBM_VBM--->STH图谱')
            s = "General_data_folder"
            print('数据处理中......')

            axis_ph(E, F, s, 0)
            Save_pH_STH(E, F, s, 0)

            x_h, x_o, sth = coSTH(0, xh, xo, s, E)
            Save_coSTH(x_h, x_o, sth, s, 0)
            coSTH_pu(0, xh, xo, s, E[-1])



            Table(D,s)
            Save_Table(D,s)




            cbm, vbm, sth = CBM_VBM(0, CBM, VBM,s,0)
            Save_CBM_VBM(cbm, vbm, sth, s,0)
            CBM_VBM_pu(0, CBM, VBM,s,E[-1])





            print('以下图普，数据以全部保存！')
            print(tabulate(table1_2, tablefmt='simple'))

            exit()
    elif ges == 2:
        VBM = float(input('请输入VBM：'))
        CBM = float(input('请输入CBM：'))
        Cha = float(input('请输入真空能级：'))
        Eg = abs(CBM - VBM)
        xh = CBM + 4.44  - Cha
        xo = -5.67 - VBM - Cha
        if xh < 0 or xo < 0:
            print("PH = 0时，不能分解水！！")
        flag1 = 0
        for ph in range(15):
            xh1 = xh - (ph * 0.059)
            xo1 = xo + (ph * 0.059)
            if xh1 >= 0 and xo1 >= 0:
                flag1 = 1
                D,E,F = General_PH(xh , xo , Eg)
                break
        if flag1 == 0:
            print('并且全PH值下都不可以分解水！')
            return
        elif flag1 == 1:
            print('在以下pH范围内，光催化材料可分解水')
            if E[0] == E[-1]:
                print("pH：", E[0], '时可分解水！')
            else:
                print("pH:", E[0], "-", E[-1])
            print(tabulate(D, headers='firstrow', tablefmt='simple', floatfmt=".2f"))
            print('数据处理中......')
            s = "General_data_folder"
            Table(D)
            Save_Table(D, s)

            x_h, x_o, sth = coSTH(0, xh, xo, s, E)
            Save_coSTH(x_h, x_o, sth, s, 0)
            coSTH_pu(0, xh, xo, s, E[-1])

            cbm, vbm, sth = CBM_VBM(0, CBM, VBM, s, 0)
            Save_CBM_VBM(cbm, vbm, sth, s, 0)
            CBM_VBM_pu(0, CBM, VBM, s, E[-1])

            axis_ph(E, F,s,0)
            Save_pH_STH(E, F, s,0)
            print('以下图谱，数据已全部保存！')
            print(tabulate(table1_2, tablefmt='simple'))

            exit()


    elif ges == 0:
        exit()
    elif ges == 9:
        return
    else:
        print("选择错误，重新输入！")
    pass
def Heterojunction_one_two():
    table1_1 = [
        ['Choose', 'Type'],
        ['1', '真空能级设为0时,输入VBM,CBM'],
        ['2', '费米能级设为0时,输入VBM,CBM,真空能级'],
        ['0', '退出'],
        ['9', '返回']
    ]
    table1_2 = [
        ['上述数据图片'],
        ['\u03c7h，\u03c7o--->STH图谱'],
        ['CBM,VBM--->STH图谱'],
        ['pH--->STH图谱'],
        ['上述数据'],
        ['\u03c7h，\u03c7o--->STH数据'],
        ['CBM,VBM--->STH数据'],
        ['pH--->STH数据'],

    ]

    table1_3 = [
        ['choose', 'type'],
        ['1', '当前PH值下的图谱'],
        ['2', '可选择PH值下的图谱']
    ]
    D = []
    E = []
    F = []
    print(tabulate(table1_1, headers='firstrow', tablefmt='simple'))
    ges = int(input("选择："))
    if ges == 1:
        VBM = float(input('请输入VBM：'))
        CBM = float(input('请输入CBM：'))
        Eg = abs(CBM - VBM)
        xh = CBM + 4.44
        xo = -5.67 - VBM
        if xh < 0 or xo < 0:
            print("PH = 0时，不能分解水！！")
        flag1 = 0
        for ph in range(15):
            xh1 = xh - (ph * 0.059)
            xo1 = xo + (ph * 0.059)
            if xh1 >= 0 and xo1 >= 0:
                flag1 = 1
                D, E, F = General_PH(xh, xo, Eg)
                break
        if flag1 == 0:
            print('并且全PH值下都不可以分解水！')
            return
        elif flag1 == 1:
            print('在以下pH范围内，光催化材料可分解水')
            if E[0] == E[-1]:
                print("pH：", E[0], '时可分解水！')
            else:
                print("pH:", E[0], "-", E[-1])
            print(tabulate(D, headers='firstrow', tablefmt='simple', floatfmt=".2f"))
            print('数据处理中......')
            s = "General_Heter_data_folder"
            Table(D, s)
            Save_Table(D, s)

            x_h, x_o, sth = coSTH(0, xh, xo, s, E)
            Save_coSTH(x_h, x_o, sth, s, 0)
            coSTH_pu(0, xh, xo, s, E[-1])

            cbm, vbm, sth = CBM_VBM(0, CBM, VBM, s, 0)
            Save_CBM_VBM(cbm, vbm, sth, s, 0)
            CBM_VBM_pu(0, CBM, VBM, s, E[-1])

            axis_ph(E, F, s,0)
            Save_pH_STH(E, F, s,0)
            print('以下图谱，数据已全部保存！')
            print(tabulate(table1_2, tablefmt='simple'))

            exit()
    elif ges == 2:
        VBM = float(input('请输入VBM：'))
        CBM = float(input('请输入CBM：'))
        Cha = float(input('请输入真空能级：'))
        Eg = abs(CBM - VBM)
        xh = CBM + 4.44 - Cha
        xo = -5.67 - VBM - Cha
        if xh < 0 or xo < 0:
            print("PH = 0时，不能分解水！！")
        flag1 = 0
        for ph in range(15):
            xh1 = xh - (ph * 0.059)
            xo1 = xo + (ph * 0.059)
            if xh1 >= 0 and xo1 >= 0:
                flag1 = 1
                D, E, F = General_PH(xh, xo, Eg)
                break
        if flag1 == 0:
            print('并且全PH值下都不可以分解水！')
            return
        elif flag1 == 1:
            print('在以下pH范围内，光催化材料可分解水')
            if E[0] == E[-1]:
                print("pH：", E[0], '时可分解水！')
            else:
                print("pH:", E[0], "-", E[-1])
            print(tabulate(D, headers='firstrow', tablefmt='simple', floatfmt=".2f"))

            print('数据处理中......')

            s = "General_Heter_data_folder"
            Table(D, s)
            Save_Table(D, s)

            x_h, x_o, sth = coSTH(0, xh, xo, s, E)
            Save_coSTH(x_h, x_o, sth, s, 0)
            coSTH_pu(0, xh, xo, s, E[-1])

            cbm, vbm, sth = CBM_VBM(0, CBM, VBM, s, 0)
            Save_CBM_VBM(cbm, vbm, sth, s, 0)
            CBM_VBM_pu(0, CBM, VBM, s, E[-1])

            axis_ph(E, F, s,0)
            Save_pH_STH(E, F, s,0)
            print('以下图谱，数据已全部保存！')
            print(tabulate(table1_2, tablefmt='simple'))
            exit()


    elif ges == 0:
        exit()
    elif ges == 9:
        return
    else:
        print("选择错误，重新输入！")


def Heterojunction_Z():
    table1_1 = [
        ['Choose', 'Type'],
        ['1', '真空能级设为0时，输入VBM,CBM'],
        ['2', '费米能级设为0时，输入VBM,CBM,真空能级'],
        ['0', '退出'],
        ['9', '返回']
    ]
    table1_2 = [
        ['上述数据图片'],
        ['\u03c7h，\u03c7o--->STH图谱'],
        ['CBM,VBM--->STH图谱'],
        ['pH--->STH图谱'],
        ['上述数据'],
        ['\u03c7h，\u03c7o--->STH数据'],
        ['CBM,VBM--->STH数据'],
        ['pH--->STH数据'],

    ]

    table1_3 = [
        ['choose', 'type'],
        ['1', '当前PH值下的图谱'],
        ['2', '可选择PH值下的图谱']
    ]

    D = []
    E = []
    F = []


    VBM1 = float(input('请输入第一个单层VBM：'))
    CBM1 = float(input('请输入第一个单层CBM：'))
    VBM2 = float(input('请输入第二个单层VBM：'))
    CBM2 = float(input('请输入第二个单层CBM：'))

    VBM = max(VBM1, VBM2)
    CBM = min(CBM1, CBM2)
    EA = np.abs(CBM1 - VBM1)
    EB = np.abs(CBM2 - VBM2)
    Eg = CBM - VBM
    xhA = CBM1 + 4.44
    xoB = -5.67 - VBM2
    xh = CBM + 4.44
    xo = -5.67 - VBM
    if xh < 0 or xo < 0:
        print("PH = 0时，不能分解水！！")
    flag1 = 0
    for ph in range(15):
        xh1 = xh - (ph * 0.059)
        xo1 = xo + (ph * 0.059)
        if xh1 >= 0 and xo1 >= 0:
            flag1 = 1
            D, E, F = Heterjunction_Z_PH(xh,xo,EA, EB, Eg)
            break
    if flag1 == 0:
        print('并且全PH值下都不可以分解水！')
        return
    elif flag1 == 1:
        print('在以下pH范围内，光催化材料可分解水')
        if E[0] == E[-1]:
            print("pH：", E[0], '时可分解水！')
        else:
            print("pH:", E[0], "-", E[-1])
        print(tabulate(D, headers='firstrow', tablefmt='simple', floatfmt=".2f"))

        print('数据处理中......')

        s = "Z_Heter_data_folder"
        Table(D, s)
        Save_Table(D, s)

        x_h, x_o, sth = coSTH_Z(0, xh, xo, s, E,EA,EB)
        Save_coSTH(x_h, x_o, sth, s, 0)
        coSTH_Z_pu(0, xh, xo, s, E[-1])

        cbm, vbm, sth = CBM_VBM_Z(0, CBM, VBM, s,0,EA,EB,xhA,xoB)
        Save_CBM_VBM(cbm, vbm, sth, s, 0)
        CBM_VBM_Z_pu(0, CBM, VBM, s)

        axis_ph(E, F, s,0)
        Save_pH_STH(E, F, s,0)
        print('以下图谱，数据已全部保存！')
        print(tabulate(table1_2, tablefmt='simple'))

        exit()








def Janus():
    table1_1 = [
        ['Choose', 'Type'],
        ['1', '真空能级设为0时,输入VBM,CBM,真空能级差'],
        ['2', '费米能级设为0时,输入VBM,CBM,真空能级,真空能级差'],
        ['0', '退出'],
        ['9', '返回']
    ]
    table1_2 = [
        ['上述数据图片'],
        ['\u03c7h，\u03c7o--->STH图谱'],
        ['CBM,VBM--->STH图谱'],
        ['pH--->STH图谱'],
        ['真空能级差,Eg--->修正STH图谱'],
        ['真空能级差--->Eg,修正STH折线图'],
        ['上述数据'],
        ['\u03c7h，\u03c7o--->STH数据'],
        ['CBM,VBM--->STH数据'],
        ['pH--->STH数据'],
        ['真空能级差,Eg--->STH数据'],
        ['真空能极差--->STH数据'],


    ]


    table1_3 = [
        ['choose', 'type'],
        ['1', '当前PH值下的图谱'],
        ['2', '可选择PH值下的图谱']
    ]
    D = []
    E = []
    F = []
    print(tabulate(table1_1, headers='firstrow', tablefmt='simple'))
    ges = int(input("选择："))
    if ges == 1:
        VBM = float(input('Please enter the VBM:'))
        CBM = float(input('Please enter the CBM:'))
        VLC = abs(float(input('Please enter the vacuum energy level difference:')))
        Eg = abs(CBM - VBM)
        xh = CBM + 4.44 + VLC
        xo = -5.67 - VBM
        if xh < 0 or xo < 0:
            print(xo)
            print("PH = 0时，不能分解水！！")
        flag1 = 0
        for ph in range(15):
            xh1 = xh - (ph * 0.059)
            xo1 = xo + (ph * 0.059)
            if xh1 >= 0 and xo1 >= 0:
                flag1 = 1
                D, E, F = Janus_PH(xh, xo, Eg,VLC)
                break
        if flag1 == 0:
            print('并且全PH值下都不可以分解水！')
            return
        elif flag1 == 1:
            print('In the following pH range, photocatalytic materials can split water')
            if E[0] == E[-1]:
                print("pH：", E[0], '时可分解水！')
            else:
                print("pH:", E[0], "-", E[-1])
            print(tabulate(D, headers='firstrow', tablefmt='simple', floatfmt=".2f"))
            print('Eg=',Eg)

            print('数据处理中......')
            s = "Janus_data_folder"

            de, eg, sth = Delta_Eg(s)
            Save_Delta_Eg(de, eg, sth, s)

            cbm, vbm, sth = CBM_VBM_J(VLC, CBM, VBM, s, 0)
            Save_CBM_VBM(cbm, vbm, sth, s, 1)
            CBM_VBM_pu(VLC, CBM, VBM, s, E[-1])

            x_h, x_o, sth = coSTH(VLC, xh, xo, s, E)
            Save_coSTH(x_h, x_o, sth, s, 1)
            coSTH_pu(VLC, xh, xo, s, E[-1])


            de, sth, eg = Zhexian_Delta_Eg_STH(s)
            Save_Zhexian_Delta_Eg_STH(de, sth, eg, s)














            axis_ph(E, F, s, 1)
            Save_pH_STH(E, F, s, 1)





            Table(D,s)
            Save_Table(D,s)











            print('以下图谱，数据已全部保存！')
            print(tabulate(table1_2, tablefmt='simple'))

            exit()

    elif ges == 2:
        VBM = float(input('请输入VBM：'))
        CBM = float(input('请输入CBM：'))
        Cha = float(input('请输入真空能级：'))
        VLC = abs(float(input('请输入Delta_phi：')))
        Eg = abs(CBM - VBM)
        xh = CBM + 4.44 + VLC - Cha
        xo = -5.67 - VBM - Cha
        if xh < 0 or xo < 0:
            print("PH = 0时，不能分解水！！")
        flag1 = 0
        for ph in range(15):
            xh1 = xh - (ph * 0.059)
            xo1 = xo + (ph * 0.059)
            if xh1 >= 0 and xo1 >= 0:
                flag1 = 1
                D, E, F = Janus_PH(xh, xo, Eg,VLC)
                break
        if flag1 == 0:
            print('并且全PH值下都不可以分解水！')
            return
        elif flag1 == 1:

            print('在以下pH范围内，光催化材料可分解水')
            if E[0] == E[-1]:
                print("pH：", E[0], '时可分解水！')
            else:
                print("pH:", E[0], "-", E[-1])
            print(tabulate(D, headers='firstrow', tablefmt='simple', floatfmt=".2f"))

            print('数据处理中......')

            s = "Janus_data_folder"

            de, sth, eg = Zhexian_Delta_Eg_STH(s)
            Save_Zhexian_Delta_Eg_STH(de, sth, eg, s)


            Table(D, s)
            Save_Table(D, s)

            x_h, x_o, sth = coSTH(VLC, xh, xo, s, E)
            Save_coSTH(x_h, x_o, sth, s, 1)
            coSTH_pu(VLC, xh, xo, s, E[-1])

            cbm, vbm, sth = CBM_VBM_J(VLC, CBM, VBM, s, 0)
            Save_CBM_VBM(cbm, vbm, sth, s, 1)
            CBM_VBM_pu(VLC, CBM, VBM, s, E[-1])

            axis_ph(E, F, s,1)
            Save_pH_STH(E, F, s,1)

            de, eg, sth = Delta_Eg(s)
            Save_Delta_Eg(de, eg, sth, s)


            print('以下图谱，数据已全部保持！')
            print(tabulate(table1_2, tablefmt='simple'))

            exit()

    elif ges == 0:
        exit()
    elif ges == 9:
        return
    else:
        print("选择错误，重新输入！")
def Janus_Heterojunction_one_two():
    table1_1 = [
        ['Choose', 'Type'],
        ['1', '真空能级设为0时,输入VBM,CBM,真空能级差'],
        ['2', '费米能级设为0时,输入VBM,CBM,真空能级,真空能级差'],
        ['0', '退出'],
        ['9', '返回']
    ]
    table1_2 = [
        ['上述数据图片'],
        ['\u03c7h，\u03c7o--->STH图谱'],
        ['CBM,VBM--->STH图谱'],
        ['pH--->STH图谱'],
        ['真空能级差,Eg--->修正STH图谱'],
        ['真空能级差--->Eg,修正STH折线图'],
        ['上述数据'],
        ['\u03c7h，\u03c7o--->STH数据'],
        ['CBM,VBM--->STH数据'],
        ['pH--->STH数据'],
        ['真空能级差,Eg--->STH数据'],
        ['真空能极差--->STH数据'],

    ]

    table1_3 = [
        ['choose', 'type'],
        ['1', '当前PH值下的图谱'],
        ['2', '可选择PH值下的图谱']
    ]
    D = []
    E = []
    F = []
    print(tabulate(table1_1, headers='firstrow', tablefmt='simple'))
    ges = int(input("选择："))
    if ges == 1:
        VBM = float(input('请输入VBM：'))
        CBM = float(input('请输入CBM：'))
        VLC = abs(float(input('请输入真空能级差：')))
        Eg = abs(CBM - VBM)
        xh = CBM + 4.44 + VLC
        xo = -5.67 - VBM
        if xh < 0 or xo < 0:
            print("PH = 0时，不能分解水！！")
        flag1 = 0
        for ph in range(15):
            xh1 = xh - (ph * 0.059)
            xo1 = xo + (ph * 0.059)
            if xh1 >= 0 and xo1 >= 0:
                flag1 = 1
                D, E, F = Janus_PH(xh, xo, Eg, VLC)
                break
        if flag1 == 0:
            print('并且全PH值下都不可以分解水！')
            return
        elif flag1 == 1:
            print('在以下pH范围内，光催化材料可分解水')
            if E[0] == E[-1]:
                print("pH：", E[0], '时可分解水！')
            else:
                print("pH:", E[0], "-", E[-1])
            print(tabulate(D, headers='firstrow', tablefmt='simple', floatfmt=".2f"))

            print('数据处理中......')

            s = "Janus_Heter_data_folder"
            Table(D, s)
            Save_Table(D, s)

            x_h, x_o, sth = coSTH(VLC, xh, xo, s, E)
            Save_coSTH(x_h, x_o, sth, s, 1)
            coSTH_pu(VLC, xh, xo, s, E[-1])

            cbm, vbm, sth = CBM_VBM_J(VLC, CBM, VBM, s, 0)
            Save_CBM_VBM(cbm, vbm, sth, s, 1)
            CBM_VBM_pu(VLC, CBM, VBM, s, E[-1])

            axis_ph(E, F, s,1)
            Save_pH_STH(E, F, s,1)

            de, eg, sth = Delta_Eg(s)
            Save_Delta_Eg(de, eg, sth, s)

            de, sth, eg = Zhexian_Delta_Eg_STH(s)
            Save_Zhexian_Delta_Eg_STH(de, sth, eg, s)
            print('以下图谱，数据已全部保存！')
            print(tabulate(table1_2, tablefmt='simple'))
            exit()

    elif ges == 2:
        VBM = float(input('请输入VBM：'))
        CBM = float(input('请输入CBM：'))
        Cha = float(input('请输入真空能级：'))
        VLC = abs(float(input('请输入Delta_phi：')))
        Eg = abs(CBM - VBM)
        xh = CBM + 4.44 + VLC - Cha
        xo = -5.67 - VBM - Cha
        if xh < 0 or xo < 0:
            print("PH = 0时，不能分解水！！")
        flag1 = 0
        for ph in range(15):
            xh1 = xh - (ph * 0.059)
            xo1 = xo + (ph * 0.059)
            if xh1 >= 0 and xo1 >= 0:
                flag1 = 1
                D, E, F = Janus_PH(xh, xo, Eg, VLC)
                break
        if flag1 == 0:
            print('并且全PH值下都不可以分解水！')
            return
        elif flag1 == 1:

            print('在以下pH范围内，光催化材料可分解水')
            if E[0] == E[-1]:
                print("pH：", E[0], '时可分解水！')
            else:
                print("pH:", E[0], "-", E[-1])
            print(tabulate(D, headers='firstrow', tablefmt='simple', floatfmt=".2f"))

            print('数据处理中......')

            s = "Janus_Heter_data_folder"
            Table(D, s)
            Save_Table(D, s)

            x_h, x_o, sth = coSTH(VLC, xh, xo, s, E)
            Save_coSTH(x_h, x_o, sth, s, 1)
            coSTH_pu(VLC, xh, xo, s, E[-1])

            cbm, vbm, sth = CBM_VBM_J(VLC, CBM, VBM, s, 0)
            Save_CBM_VBM(cbm, vbm, sth, s, 1)
            CBM_VBM_pu(VLC, CBM, VBM, s, E[-1])

            axis_ph(E, F, s,1)
            Save_pH_STH(E, F, s,1)

            de, eg, sth = Delta_Eg(s)
            Save_Delta_Eg(de, eg, sth, s)

            de, sth, eg = Zhexian_Delta_Eg_STH(s)
            Save_Zhexian_Delta_Eg_STH(de, sth, eg, s)
            print('以下图谱，数据已全部保存！')
            print(tabulate(table1_2, tablefmt='simple'))
            exit()

    elif ges == 0:
        exit()
    elif ges == 9:
        return
    else:
        print("选择错误，重新输入！")
def Janus_Z():
    table1_1 = [
        ['Choose', 'Type'],
        ['0', '退出'],
        ['9', '返回']
    ]
    table1_2 = [
        ['上述数据图片'],
        ['\u03c7h，\u03c7o--->STH图谱'],
        ['CBM,VBM--->STH图谱'],
        ['pH--->STH图谱'],
        ['真空能级差,Eg--->修正STH图谱'],
        ['真空能级差--->Eg,修正STH折线图'],
        ['上述数据'],
        ['\u03c7h，\u03c7o--->STH数据'],
        ['CBM,VBM--->STH数据'],
        ['pH--->STH数据'],
        ['真空能级差,Eg--->STH数据'],
        ['真空能极差--->STH数据'],
    ]

    table1_3 = [
        ['choose', 'type'],
        ['1', '当前PH值下的图谱'],
        ['2', '可选择PH值下的图谱']
    ]
    D = []
    E = []
    F = []
    # print(tabulate(table1_2, headers='firstrow', tablefmt='simple'))
    # VBM1 = float(input('请输入第一个单层VBM：'))
    # CBM1 = float(input('请输入第一个单层CBM：'))
    # VBM2 = float(input('请输入第二个单层VBM：'))
    # CBM2 = float(input('请输入第二个单层CBM：'))
    VLC = abs(float(input('请输入真空能级差：')))
    Eg1 = float(input('输入Eg1:'))
    Eg2 = float(input('输入Eg2:'))
    xh = float(input('输入xh:'))
    xo = float(input('输入xo:'))
    # VBM = min(VBM1, VBM2)
    # CBM = max(CBM1, CBM2)
    # Eg = abs(CBM - VBM)
    # xh = CBM + 4.44 + VLC
    # xo = -5.67 - VBM
    Eg = max(Eg1,Eg2)



    print('Eg=',Eg)

    flag1 = 0
    for ph in range(15):
        xh1 = xh - (ph * 0.059)
        xo1 = xo + (ph * 0.059)
        if xh1 >= 0 and xo1 >= 0:
            flag1 = 1
            D, E, F = Janus_Z_PH(xh, xo, Eg,VLC)
    if flag1 == 0:
        print('并且全PH值下都不可以分解水！')
        return
    elif flag1 == 1:
        print('在以下pH范围内，光催化材料可分解水')
        if E[0] == E[-1]:
            print("pH：", E[0], '时可分解水！')
        else:
            print("pH:", E[0], "-", E[-1])
        print(tabulate(D, headers='firstrow', tablefmt='simple', floatfmt=".2f"))

        # print('数据处理中......')
        s = "Z_Janus_data_folder"

        # de, eg, sth = Delta_Eg_STH_Z(s)
        # Save_Delta_Eg(de, eg, sth, s)
        #
        # de, sth, eg = Zhexian_Delta_Eg_STH_Z(s)
        # Save_Zhexian_Delta_Eg_STH(de, sth, eg, s)
        # Table(D, s)
        # Save_Table(D, s)
        #
        # x_h, x_o, sth = coSTH_Z_J(VLC, xh, xo, s,E)
        # Save_coSTH(x_h, x_o, sth, s, 1)
        # coSTH_Z_pu(VLC, xh, xo, s, E[-1])

        # cbm, vbm, sth = CBM_VBM_Z_J(VLC, CBM, VBM, s, 0)
        # Save_CBM_VBM(cbm, vbm, sth, s, 1)
        # CBM_VBM_Z_pu(VLC, CBM, VBM, s)

        axis_ph(E, F, s,1)
        Save_pH_STH(E, F, s,1)




        print('以下图谱，数据已全部保存！')
        print(tabulate(table1_2, tablefmt='simple'))
        exit()
        # elif ges == 0:
        #     exit()
        # elif ges == 9:
        #     return
        # else:
        #     print("选择错误，重新输入！")
    pass
while True:
    table0 = [
        ['Choose', 'Type'],
        ['1', '单层'],
        ['2', '二维Janus单层'],
        ['3', 'Ⅰ型或Ⅱ型异质结'],
        ['4', 'Ⅰ型或Ⅱ型Janus异质结'],
        ['5', 'Z型异质结'],
        ['6', 'Z型Janus异质结'],
        ['0', '退出！']
    ]
    table1 = [
        ['choose', 'type'],
        ['1', 'Xh，Xo--->修正STH'],
        ['2', 'Delta，Eg--->极限修正STH'],
        ['3', 'Eg--->修正STH折线图'],
        ['4', 'Delta--->Eg,极限修正STH'],
        ['0', '退出'],
        ['9', '返回']
    ]
    table2 = [
        ['choose', 'type'],
        ['1', 'Xh，Xo--->STH'],
        ['3', 'Eg--->修正STH折线图'],
        ['0', '退出'],
        ['9', '返回']
    ]
    # print('H\u2082O')  # 输出 H₂O
    # print('S\u2095T\u209cH\u2092')  # 输出 SₕTₜHₒ
    print(tabulate(table0, headers='firstrow', tablefmt='simple'))
    ges = int(input("选择："))
    if ges == 1:
        General()

    elif ges == 2:

        Janus()

    elif ges == 3:

        Heterojunction_one_two()

        pass

    elif ges == 4:
        Janus_Heterojunction_one_two()
        pass
    elif ges == 5:
        Heterojunction_Z()
        pass
    elif ges == 6:
        Janus_Z()
    elif ges == 0:
        break
        pass




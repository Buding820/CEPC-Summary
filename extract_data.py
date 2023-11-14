#!/usr/bin/env python3 

import h5py 
import numpy as np 
import copy 
import pandas as pd
import os, sys 
import matplotlib.pyplot as plt 
from mpl_toolkits.axisartist.axislines import AxesZero
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

config = {
    "font.family":  'serif',
    "font.serif": 'stix',
    "mathtext.fontset": "stix"
}
plt.rcParams.update(config)

pwd = os.path.abspath(os.path.dirname(__file__))

# Computes the elementwise minimum of the input arrays
def bulk_min(x, limit):
  indices = np.where(x > limit)
  z = copy.deepcopy(x)
  z[indices] = limit
  return z

# Read H5DF file 
def read_h5py_to_csv(h5, nm, out, keys):
    with h5py.File(h5, "r") as f1:
        df = f1[nm]
        # with open("NUHM2-keys.dat", 'w') as f1:
        #     for kk in df.keys():
        #         # f1.write("NUHM2/{}\n".format(kk))

        data = {}
        for kk, vv in keys.items():
            # print(kk, min(np.array(df[vv])), max(np.array(df[vv])))
            data[kk] = np.array(df[vv])

        ds = pd.DataFrame(data)
        ds = ds.sort_values(['LogLike'], ascending=False)
        ds = ds[ds['isvalid'] == 1]
        
        # ds['LogLike'] -= ds['LogLike'] 
        #                 +ds['LogLike']
        print(ds.shape)
#        ds.to_csv("NUHM2/{}.csv".format(out))
        return ds 

def draw_ax_violin(df, ax, var, bwl=0.05, bwr=0.05, raw=False):
    """
    bwl, bwr :  A float to modify the gaussian smearing width, it should be smaller enough,
                However, if it is too small, the distribution will be like zigzag.
    raw      :  Need to be False when public the plot.
    Good luck !
    """


    print("===== Plotting =====\n\t{}".format(var['name']))
    # ax.axis("off")
    ax.set_facecolor("None")
    xx = np.linspace(var['range'][0], var['range'][1], var['nbin'])
    ds = pd.DataFrame({"xx" : (xx[:-1] + xx[1:]) / 2})
    dx = (var['range'][1] - var['range'][0]) / (var['nbin'] + 1) 

    ds = pd.DataFrame({
        "xi": ds['xx'],
        "PLo": ds.apply(lambda tt: df[(df[var['name']] >= tt['xx'] - dx) & (df[var['name']] <= tt['xx'] + dx)].LogLike.max(axis=0, skipna=True), axis=1 ),
        "PLn": ds.apply(lambda tt: df[(df[var['name']] >= tt['xx'] - dx) & (df[var['name']] <= tt['xx'] + dx)].NewLogL.max(axis=0, skipna=True), axis=1 )
    }).fillna({"PLo": -np.inf, "PLn": -np.inf})
    from scipy.stats import gaussian_kde
    bso  = ds['PLo'].max()
    bsn  = ds['PLn'].max()

    PLo_KDE = gaussian_kde((ds['xi'] - var['range'][0])/ (var['range'][1] - var['range'][0]), bw_method=bwl, weights=np.exp(ds['PLo'])/np.exp(bso))
    PLn_KDE = gaussian_kde((ds['xi'] - var['range'][0])/ (var['range'][1] - var['range'][0]), bw_method=bwr, weights=np.exp(ds['PLn'])/np.exp(bsn))

    xi = np.linspace(0.0, 1.0, 1000)
    x0 = np.zeros(1000)
    plokde = PLo_KDE(xi)
    plokde = plokde / max(plokde)
    plnkde = PLn_KDE(xi)
    plnkde = plnkde / max(plnkde)

    if raw:
        ax.scatter(-np.exp(ds['PLo'])/np.exp(bso), ds['xi'], marker='o', color="red", s=2)
        ax.scatter(np.exp(ds['PLn'])/np.exp(bsn), ds['xi'], marker='o', color="black", s=2)

    ax.fill_betweenx((var['range'][1] - var['range'][0])*xi + var['range'][0], x0, -plokde,  fc="#E64980", alpha=0.8, zorder=10)
    ax.fill_betweenx((var['range'][1] - var['range'][0])*xi + var['range'][0], x0, plnkde,  fc="#1098AD", alpha=0.8, zorder=10)
    # ax.plot(-plokde, (var['range'][1] - var['range'][0])*xi + var['range'][0], c='orange', zorder=0)
    # ax.plot(plnkde, (var['range'][1] - var['range'][0])*xi + var['range'][0],  c='green', zorder=0)
    
    for direction in ["right", "bottom", "top"]:
        # hides borders
        ax.spines[direction].set_visible(False)
    ax.spines["left"].set_position(("data", 0))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', labelsize=8, direction="inout")
    ax.tick_params(which='major', length=8, width=1.)
    ax.tick_params(which='minor', length=3.5, width=1.)
    # ax.xaxis.set_visible(False)
    ax.xaxis.set_ticks([])
    ax.set_ylim(var['range'])
    ax.set_xlim(-1.01, 1.01)
    ax.set_xlabel(var['label'], fontsize=12)
    
    
    # ax.yaxis.set_visible(True)

if __name__ == "__main__":
    switch_cmssm = True
    switch_nuhm1 = True
    switch_nuhm2 = True
    switch_mssm7 = True

    # ============== H5DF path and informations ============== #

    dtpth_cmssm = "../CMSSM_recal_2023.hdf5" #/cmssm_data
    dtpth_nuhm1 = "../NUHM1_recal_2023.hdf5" #/nuhm1_data
    dtpth_nuhm2 = "../NUHM2_recal_2023.hdf5" #/nuhm2_data
    dtpth_mssm7 = "../MSSM7_recal_2023.hdf5" #/MSSM7_data

    cmssm_info = { # this is CMSSM !! no mHh mhu !!!!!!!  -- ----- b2sgamma_LL2
            "A0":           "#CMSSM_parameters @CMSSM::primary_parameters::A0",
            "M0":           "#CMSSM_parameters @CMSSM::primary_parameters::M0",
            "M12":          "#CMSSM_parameters @CMSSM::primary_parameters::M12",
            "SignMu":       "#CMSSM_parameters @CMSSM::primary_parameters::SignMu",
            "TanBeta":      "#CMSSM_parameters @CMSSM::primary_parameters::TanBeta",
            "LHC_1_raw":    "#LHC_Combined_LogLike @ColliderBit::calc_LHC_LogLike RunI",
            "LHC_2_raw":    "#LHC_Combined_LogLike @ColliderBit::calc_LHC_LogLike",
            "SL_LL":        "#SL_LL @FlavBit::SL_likelihood buggy",
            "b2ll_LL":      "#b2ll_LL @FlavBit::b2ll_likelihood buggy",
            "b2sll_LL":     "#b2sll_LL @FlavBit::b2sll_likelihood buggy",
            "b2sgamma_LL":  "#b2sgamma_LL @FlavBit::b2sgamma_likelihood buggy",
            "b2sgamma_LL2": "#b2sgamma_LL @FlavBit::b2sgamma_likelihood still buggy",
            "LogLike":      "LogLike",
            "isvalid":      "#CEPC_Higgs_Couplings @ColliderBit::get_CEPC_Higgs_Couplings::Mh_SM_like_isvalid",  
            "CEPC_Higgs_LogLike": "#CEPC_Higgs_SMlike_LogLike @ColliderBit::calc_CEPC_Higgs_SMlike_LogLike"
        }

    numh1_info = { # this is nuhm1 !! no mHd
            "A0":           "#NUHM1_parameters @NUHM1::primary_parameters::A0",
            "M0":           "#NUHM1_parameters @NUHM1::primary_parameters::M0",
            "M12":          "#NUHM1_parameters @NUHM1::primary_parameters::M12",
            "SignMu":       "#NUHM1_parameters @NUHM1::primary_parameters::SignMu",
            "TanBeta":      "#NUHM1_parameters @NUHM1::primary_parameters::TanBeta",
            "mH":           "#NUHM1_parameters @NUHM1::primary_parameters::mH",
            "LHC_1_raw":    "#LHC_Combined_LogLike @ColliderBit::calc_LHC_LogLike RunI",
            "LHC_2_raw":    "#LHC_Combined_LogLike @ColliderBit::calc_LHC_LogLike",
            "SL_LL":        "#SL_LL @FlavBit::SL_likelihood buggy",
            "b2ll_LL":      "#b2ll_LL @FlavBit::b2ll_likelihood buggy",
            "b2sll_LL":     "#b2sll_LL @FlavBit::b2sll_likelihood buggy",
            "b2sgamma_LL":  "#b2sgamma_LL @FlavBit::b2sgamma_likelihood buggy",
            "LogLike":      "LogLike",
            "isvalid":      "LogLike_isvalid",  
            "CEPC_Higgs_LogLike": "#CEPC_Higgs_SMlike_LogLike @ColliderBit::calc_CEPC_Higgs_SMlike_LogLike"
       }
        
    numh2_info = {
            "A0":           "#NUHM2_parameters @NUHM2::primary_parameters::A0",
            "M0":           "#NUHM2_parameters @NUHM2::primary_parameters::M0",
            "M12":          "#NUHM2_parameters @NUHM2::primary_parameters::M12",
            "SignMu":       "#NUHM2_parameters @NUHM2::primary_parameters::SignMu",
            "TanBeta":      "#NUHM2_parameters @NUHM2::primary_parameters::TanBeta",
            "mHu":          "#NUHM2_parameters @NUHM2::primary_parameters::mHu",
            "mHd":          "#NUHM2_parameters @NUHM2::primary_parameters::mHd",
            "LHC_1_raw":    "#LHC_Combined_LogLike @ColliderBit::calc_LHC_LogLike RunI",
            "LHC_2_raw":    "#LHC_Combined_LogLike @ColliderBit::calc_LHC_LogLike",
            "SL_LL":        "#SL_LL @FlavBit::SL_likelihood buggy",
            "b2ll_LL":      "#b2ll_LL @FlavBit::b2ll_likelihood buggy",
            "b2sll_LL":     "#b2sll_LL @FlavBit::b2sll_likelihood buggy",
            "b2sgamma_LL":  "#b2sgamma_LL @FlavBit::b2sgamma_likelihood buggy",
            "LogLike":      "LogLike",
            "isvalid":      "LogLike_isvalid",  
            "CEPC_Higgs_LogLike": "#CEPC_Higgs_SMlike_LogLike @ColliderBit::calc_CEPC_Higgs_SMlike_LogLike"
       }
    
    mssm7_info = { # this is MSSM7 
            "Ad3":          "#MSSM7atQ_parameters @MSSM7atQ::primary_parameters::Ad_3",
            "Au3":          "#MSSM7atQ_parameters @MSSM7atQ::primary_parameters::Au_3",
            "M2":           "#MSSM7atQ_parameters @MSSM7atQ::primary_parameters::M2",
            "SignMu":       "#MSSM7atQ_parameters @MSSM7atQ::primary_parameters::SignMu",
            "TanBeta":      "#MSSM7atQ_parameters @MSSM7atQ::primary_parameters::TanBeta",
            "mHu2":         "#MSSM7atQ_parameters @MSSM7atQ::primary_parameters::mHu2",
            "mHd2":         "#MSSM7atQ_parameters @MSSM7atQ::primary_parameters::mHd2",
            "mf2":          "#MSSM7atQ_parameters @MSSM7atQ::primary_parameters::mf2",
            "LHC_1_raw":    "#LHC_Combined_LogLike @ColliderBit::calc_LHC_LogLike RunI",
            "LHC_2_raw":    "#LHC_Combined_LogLike @ColliderBit::calc_LHC_LogLike",
            "SL_LL":        "#SL_LL @FlavBit::SL_likelihood buggy",
            "b2ll_LL":      "#b2ll_LL @FlavBit::b2ll_likelihood buggy",
            "b2sll_LL":     "#b2sll_LL @FlavBit::b2sll_likelihood buggy",
            "b2sgamma_LL":  "#b2sgamma_LL @FlavBit::b2sgamma_likelihood buggy",
            "LogLike":      "LogLike",
            "isvalid":      "#CEPC_Higgs_Couplings @ColliderBit::get_CEPC_Higgs_Couplings::Mh_SM_like_isvalid",  
            "CEPC_Higgs_LogLike": "#CEPC_Higgs_SMlike_LogLike @ColliderBit::calc_CEPC_Higgs_SMlike_LogLike"
        }

    # fig = plt.figure(figsize=(12, 16))
    fig = plt.figure(figsize=(9, 12))
    
    # =============== CMSSM ================ #
    if switch_cmssm:
        df = read_h5py_to_csv(dtpth_cmssm, "CMSSM", "CMSSMplot", cmssm_info)
        df['LHC_1'] = df['LHC_1_raw'] - bulk_min(np.array(df['LHC_1_raw']), 0)
        df['LHC_2'] = df['LHC_2_raw'] - bulk_min(np.array(df['LHC_2_raw']), 0)
        df['LogLike'] = df['LogLike'] - df['LHC_1'] - df['LHC_2'] - df['SL_LL'] - df['b2ll_LL'] - df['b2sll_LL'] -df['b2sgamma_LL'] -df['b2sgamma_LL2'] - df['CEPC_Higgs_LogLike'] #for CMSSM
        df['NewLogL'] = df["LogLike"] + df['CEPC_Higgs_LogLike']

        ax1   = fig.add_axes([0.0,  0.7525,  1.0,  0.2375])
        ax1.axis("off")
        ax11  = fig.add_axes([0.07, 0.7900, 0.13, 0.2])
        ax12  = fig.add_axes([0.20, 0.7900, 0.13, 0.2])
        ax13  = fig.add_axes([0.33, 0.7900, 0.13, 0.2])
        ax14  = fig.add_axes([0.46, 0.7900, 0.13, 0.2])

        vinf_cmssm = [
            {
                "name": "M0",
                "range": [0, 10000],
                "nbin": 100,
                "label": r"$m_0~({\rm GeV})$"
            },
            {
                "name": "A0",
                "range": [-10000, 10000],
                "nbin": 100,
                "label": r"$A_0~({\rm GeV})$"
            },
            {
                "name": "M12",
                "range": [0, 10000],
                "nbin": 100,
                "label": r"$m_{1/2}~({\rm GeV})$"
            },      
            {
                "name": "TanBeta",
                "range": [0, 60],
                "nbin": 100,
                "label": r"$\tan{\beta}$"
            }
        ]

        draw_ax_violin(df, ax11, vinf_cmssm[0], bwl=0.05, bwr=0.3, raw=False)
        draw_ax_violin(df, ax12, vinf_cmssm[1], bwl=0.05, bwr=0.3, raw=False)
        draw_ax_violin(df, ax13, vinf_cmssm[2], bwl=0.05, bwr=0.4, raw=False)
        draw_ax_violin(df, ax14, vinf_cmssm[3], bwl=0.05, bwr=0.3, raw=False)

        ax1.text(0.04, 0.75,  r"$\rm CMSSM$", fontsize=16, transform=ax1.transAxes, ha="left", va='top', rotation=90)

        df = None

    # =============== HUHM1 ================ #
    if switch_nuhm1:
        df = read_h5py_to_csv(dtpth_nuhm1, "NUHM1", "NUHM1plot", numh1_info)

        df['LHC_1'] = df['LHC_1_raw'] - bulk_min(np.array(df['LHC_1_raw']), 0)
        df['LHC_2'] = df['LHC_2_raw'] - bulk_min(np.array(df['LHC_2_raw']), 0)
        df['LogLike'] = df['LogLike'] - df['LHC_1'] - df['LHC_2'] - df['SL_LL'] - df['b2ll_LL'] - df['b2sll_LL'] -df['b2sgamma_LL'] - df['CEPC_Higgs_LogLike'] # for NUHM12
        df['NewLogL'] = df["LogLike"] + df['CEPC_Higgs_LogLike'] 

        ax2   = fig.add_axes([0.0,  0.505,  1.0,  0.2375])
        ax2.axis("off")

        ax21  = fig.add_axes([0.07, 0.5425, 0.13, 0.2])
        ax22  = fig.add_axes([0.20, 0.5425, 0.13, 0.2])
        ax23  = fig.add_axes([0.33, 0.5425, 0.13, 0.2])
        ax24  = fig.add_axes([0.46, 0.5425, 0.13, 0.2])
        ax25  = fig.add_axes([0.59, 0.5425, 0.13, 0.2])

        vinf_numh1 = [
            {
                "name": "M0",
                "range": [0, 10000],
                "nbin": 100,
                "label": r"$m_0~({\rm GeV})$"
            },
            {
                "name": "A0",
                "range": [-10000, 10000],
                "nbin": 100,
                "label": r"$A_0~({\rm GeV})$"
            },
            {
                "name": "M12",
                "range": [0, 10000],
                "nbin": 100,
                "label": r"$m_{1/2}~({\rm GeV})$"
            },
            {
                "name": "mH",
                "range": [0, 10000],
                "nbin": 100,
                "label": r"$m_{H_{u/d}}~({\rm GeV})$"
            },               
            {
                "name": "TanBeta",
                "range": [0, 60],
                "nbin": 100,
                "label": r"$\tan{\beta}$"
            }
        ]

        draw_ax_violin(df, ax21, vinf_numh1[0], bwl=0.05, bwr=0.05, raw=False)
        draw_ax_violin(df, ax22, vinf_numh1[1], bwl=0.05, bwr=0.05, raw=False)
        draw_ax_violin(df, ax23, vinf_numh1[2], bwl=0.05, bwr=0.1, raw=False)
        draw_ax_violin(df, ax24, vinf_numh1[3], bwl=0.05, bwr=0.1, raw=False)
        draw_ax_violin(df, ax25, vinf_numh1[4], bwl=0.05, bwr=0.07, raw=False)

        ax2.text(0.04, 0.75,  r"$\rm NUHM1$", fontsize=16, transform=ax2.transAxes, ha="left", va='top', rotation=90)


        df = None

    # =============== HUHM2 ================ #
    if switch_nuhm2:
        df = read_h5py_to_csv(dtpth_nuhm2, "NUHM2", "NUHM2plot", numh2_info)
        df['LHC_1'] = df['LHC_1_raw'] - bulk_min(np.array(df['LHC_1_raw']), 0)
        df['LHC_2'] = df['LHC_2_raw'] - bulk_min(np.array(df['LHC_2_raw']), 0)
        df['LogLike'] = df['LogLike'] - df['LHC_1'] - df['LHC_2'] - df['SL_LL'] - df['b2ll_LL'] - df['b2sll_LL'] -df['b2sgamma_LL'] - df['CEPC_Higgs_LogLike'] 
        df['NewLogL'] = df["LogLike"] + df['CEPC_Higgs_LogLike'] 

        ax3   = fig.add_axes([0.0,  0.2575,  1.0,  0.2375])
        ax3.axis("off")
        ax31  = fig.add_axes([0.07, 0.295, 0.13, 0.2])
        ax32  = fig.add_axes([0.20, 0.295, 0.13, 0.2])
        ax33  = fig.add_axes([0.33, 0.295, 0.13, 0.2])
        ax34  = fig.add_axes([0.46, 0.295, 0.13, 0.2])
        ax35  = fig.add_axes([0.59, 0.295, 0.13, 0.2])
        ax36  = fig.add_axes([0.72, 0.295, 0.13, 0.2])

        vinf_numh2 = [
            {
                "name": "M0",
                "range": [0, 10000],
                "nbin": 100,
                "label": r"$m_0~({\rm GeV})$"
            },
            {
                "name": "A0",
                "range": [-10000, 10000],
                "nbin": 100,
                "label": r"$A_0~({\rm GeV})$"
            },
            {
                "name": "M12",
                "range": [0, 10000],
                "nbin": 100,
                "label": r"$m_{1/2}~({\rm GeV})$"
            },
            {
                "name": "mHu",
                "range": [0, 10000],
                "nbin": 100,
                "label": r"$m_{H_{u}}~({\rm GeV})$"
            },        
            {
                "name": "mHd",
                "range": [0, 10000],
                "nbin": 100,
                "label": r"$m_{H_d}~({\rm GeV})$"
            },        
            {
                "name": "TanBeta",
                "range": [0, 60],
                "nbin": 100,
                "label": r"$\tan{\beta}$"
            }
        ]

        draw_ax_violin(df, ax31, vinf_numh2[0], bwl=0.05, bwr=0.08, raw=False)
        draw_ax_violin(df, ax32, vinf_numh2[1], bwl=0.03, bwr=0.10, raw=False)
        draw_ax_violin(df, ax33, vinf_numh2[2], bwl=0.03, bwr=0.14, raw=False)
        draw_ax_violin(df, ax34, vinf_numh2[3], bwl=0.04, bwr=0.06, raw=False)
        draw_ax_violin(df, ax35, vinf_numh2[4], bwl=0.04, bwr=0.10, raw=False)
        draw_ax_violin(df, ax36, vinf_numh2[5], bwl=0.025, bwr=0.08, raw=False)

        ax3.text(0.04, 0.75, r"$\rm NUHM2$", fontsize=16, transform=ax3.transAxes, ha="left", va='top', rotation=90)

        df = None

    # =============== MSSM7 ================ #
    if switch_mssm7:
        df = read_h5py_to_csv(dtpth_mssm7, "MSSM7", "MSSM7plot", mssm7_info)
        df['LHC_1'] = df['LHC_1_raw'] - bulk_min(np.array(df['LHC_1_raw']), 0)
        df['LHC_2'] = df['LHC_2_raw'] - bulk_min(np.array(df['LHC_2_raw']), 0)
        df['LogLike'] = df['LogLike'] - df['LHC_1'] - df['LHC_2'] - df['SL_LL'] - df['b2ll_LL'] - df['b2sll_LL'] -df['b2sgamma_LL'] - df['CEPC_Higgs_LogLike'] # for NUHM12
        df['NewLogL'] = df['LogLike'] + df['CEPC_Higgs_LogLike'] # for NUHM12

        ax4   = fig.add_axes([0.0,  0.01,  1.0, 0.2375])
        ax4.axis("off")
        ax41  = fig.add_axes([0.07, 0.0475, 0.13, 0.2])
        ax42  = fig.add_axes([0.20, 0.0475, 0.13, 0.2])
        ax43  = fig.add_axes([0.33, 0.0475, 0.13, 0.2])
        ax44  = fig.add_axes([0.46, 0.0475, 0.13, 0.2])
        ax45  = fig.add_axes([0.59, 0.0475, 0.13, 0.2])
        ax46  = fig.add_axes([0.72, 0.0475, 0.13, 0.2])
        ax47  = fig.add_axes([0.85, 0.0475, 0.13, 0.2])

        vinf_mssm7 = [
            {
                "name": "Ad3",
                "range": [0, 10000],
                "nbin": 100,
                "label": r"$A_{d3}~({\rm GeV})$"
            },
            {
                "name": "Au3",
                "range": [-10000, 10000],
                "nbin": 100,
                "label": r"$A_{u3}~({\rm GeV})$"
            },
            {
                "name": "M2",
                "range": [0, 10000],
                "nbin": 100,
                "label": r"$m_{2}~({\rm GeV})$"
            },
            {
                "name": "mHu2",
                "range": [-100000000, 100000000],
                "nbin": 100,
                "label": r"$m_{H_u}^2~({\rm GeV}^2)$"
            },        
            {
                "name": "mHd2",
                "range": [-100000000, 100000000],
                "nbin": 100,
                "label": r"$m_{H_d}^2~({\rm GeV}^2)$"
            },        
            {
                "name": "mf2",
                "range": [0, 100000000],
                "nbin": 100,
                "label": r"$m_f^2~({\rm GeV})^2$"
            },
            {
                "name": "TanBeta",
                "range": [0, 60],
                "nbin": 100,
                "label": r"$\tan{\beta}$"
            },
        ]

        draw_ax_violin(df, ax41, vinf_mssm7[0], bwl=0.05, bwr=0.05, raw=False)
        draw_ax_violin(df, ax42, vinf_mssm7[1], bwl=0.05, bwr=0.05, raw=False)
        draw_ax_violin(df, ax43, vinf_mssm7[2], bwl=0.05, bwr=0.05, raw=False)
        draw_ax_violin(df, ax44, vinf_mssm7[3], bwl=0.20, bwr=0.20, raw=False)
        draw_ax_violin(df, ax45, vinf_mssm7[4], bwl=0.05, bwr=0.05, raw=False)
        draw_ax_violin(df, ax46, vinf_mssm7[5], bwl=0.05, bwr=0.05, raw=False)
        draw_ax_violin(df, ax47, vinf_mssm7[6], bwl=0.05, bwr=0.05, raw=False)
        
        ax4.text(0.04, 0.75, r"$\rm MSSM7$", fontsize=16, transform=ax4.transAxes, ha="left", va='top', rotation=90)
    
    
#    plt.show()
    plt.savefig("Summary.png", dpi=300)




#!/usr/bin/env python

import h5py
import numpy as np
import copy
import matplotlib.pyplot as plt

# Computes the elementwise minimum of the input arrays
def bulk_min(x, limit):
  indices = np.where(x > limit)
  z = copy.deepcopy(x)
  z[indices] = limit
  return z

def chisq_fun(mu, err1, err2, mean=1):
  return pow(mu-mean,2) / (pow(err1,2)+pow(err2,2))

def CEPC_loglike(Mh_SM_like, BR_hjcc, BR_hjbb, BR_hjmumu, BR_hjtautau, BR_hjWW, \
                BR_hjZZ, BR_hjgaga, BR_hjgg, CS_Z, CS_W):
  #MSSM7-BestFit
  mh_SM =  124.75458835041347
  BR_hjbb_SM =  0.6209027694151499
  BR_hjcc_SM =  0.03269012046653416
  BR_hjgg_SM =  0.0653613356224161
  BR_hjWW_SM =  0.19165075978002685
  BR_hjtautau_SM =  0.06100444688640195
  BR_hjZZ_SM =  0.023993319577253393
  BR_hjgaga_SM =  0.002531955206650806
  BR_hjmumu_SM =  0.00021597793230409233
  CS_Z_SM =  0.9767358061178618
  CS_W_SM =  0.9563672855431476


#  # SM-like
#  mh_SM = 125.25
#  BR_hjbb_SM = 0.577
#  BR_hjcc_SM = 0.0291
#  BR_hjgg_SM = 0.0857
#  BR_hjWW_SM = 0.215
#  BR_hjtautau_SM = 0.0632
#  BR_hjZZ_SM = 0.0264
#  BR_hjgaga_SM = 0.00228
#  BR_hjmumu_SM = 2.19e-04
#  # SM Higgs normalized cross section
#  CS_Z_SM = 1.
#  CS_W_SM = 1.
#  chisq_mh = chisq_fun(Mh_SM_like, 2., 0.017 , mh_SM)
  
  
  chisq_mu=0.0

  CEPC = True
  FCCee = False
  ILC =False

  k_Z = CS_Z/CS_Z_SM
  K_W = CS_W/CS_W_SM
  if CEPC:
    chisq_mu = ( chisq_fun( k_Z, 0.5/100., 0)  
           + chisq_fun( k_Z* BR_hjbb / BR_hjbb_SM ,  0.27/100., 3.3/100.)  #h->bb channel
           + chisq_fun( k_Z* BR_hjcc / BR_hjcc_SM,   3.3/100., 12./100.)  #h->cc channel
           + chisq_fun( k_Z* BR_hjgg / BR_hjgg_SM,   1.3/100., 10./100.)  #h->gg channel
           + chisq_fun( k_Z* BR_hjWW / BR_hjWW_SM,   1.0/100., 4.3/100.)  #h->WW channel
           + chisq_fun( k_Z* BR_hjtautau/ BR_hjtautau_SM,   0.8/100., 5.7/100.)  #h->tautau channel
           + chisq_fun( k_Z* BR_hjZZ / BR_hjZZ_SM,   5.1/100., 4.3/100.)  #h->ZZ channel
           + chisq_fun( k_Z* BR_hjgaga / BR_hjgaga_SM,   6.8/100., 5.0/100.)  #h->gaga channel
           + chisq_fun( k_Z* BR_hjmumu / BR_hjmumu_SM,   17.0/100., 6.0/100.)  #h->mumu channel
           + chisq_fun( K_W* BR_hjbb / BR_hjbb_SM, 2.8/100., 3.3/100.) ) #(nunu)h->bb channel
  elif FCCee:
    # 240GeV Zh production channel
    chisq_mu = ( chisq_fun( k_Z, 0.5/100., 0)  
           + chisq_fun( k_Z* BR_hjbb / BR_hjbb_SM ,  0.3/100., 3.3/100.)  #h->bb channel
           + chisq_fun( k_Z* BR_hjcc / BR_hjcc_SM,   2.2/100., 12./100.)  #h->cc channel
           + chisq_fun( k_Z* BR_hjgg / BR_hjgg_SM,   1.9/100., 10./100.)  #h->gg channel
           + chisq_fun( k_Z* BR_hjWW / BR_hjWW_SM,   1.2/100., 4.3/100.)  #h->WW channel
           + chisq_fun( k_Z* BR_hjtautau/ BR_hjtautau_SM,   0.9/100., 5.7/100.)  #h->tautau channel
           + chisq_fun( k_Z* BR_hjZZ / BR_hjZZ_SM,   4.4/100., 4.3/100.)  #h->ZZ channel
           + chisq_fun( k_Z* BR_hjgaga / BR_hjgaga_SM,   9.0/100., 5.0/100.)  #h->gaga channel
           + chisq_fun( k_Z* BR_hjmumu / BR_hjmumu_SM,   19.0/100., 6.0/100.)  #h->mumu channel
           + chisq_fun( K_W* BR_hjbb / BR_hjbb_SM, 3.1/100., 3.3/100.) ) #(nunu)h->bb channel
    # 365GeV Zh production channel
    chisq_mu += ( chisq_fun( k_Z, 0.9/100., 0)  
           + chisq_fun( k_Z* BR_hjbb / BR_hjbb_SM ,  0.5/100., 3.3/100.)  #h->bb channel
           + chisq_fun( k_Z* BR_hjcc / BR_hjcc_SM,   6.5/100., 12./100.)  #h->cc channel
           + chisq_fun( k_Z* BR_hjgg / BR_hjgg_SM,   3.5/100., 10./100.)  #h->gg channel
           + chisq_fun( k_Z* BR_hjWW / BR_hjWW_SM,   2.6/100., 4.3/100.)  #h->WW channel
           + chisq_fun( k_Z* BR_hjtautau/ BR_hjtautau_SM,   1.8/100., 5.7/100.)  #h->tautau channel
           + chisq_fun( k_Z* BR_hjZZ / BR_hjZZ_SM,   12./100., 4.3/100.)  #h->ZZ channel
           + chisq_fun( k_Z* BR_hjgaga / BR_hjgaga_SM,   18./100., 5.0/100.)  #h->gaga channel
           + chisq_fun( k_Z* BR_hjmumu / BR_hjmumu_SM,   40.0/100., 6.0/100.) ) #h->mumu channel
    # 365GeV nunuh production channel
    chisq_mu += (  
           + chisq_fun( k_Z* BR_hjbb / BR_hjbb_SM ,  0.9/100., 3.3/100.)  #h->bb channel
           + chisq_fun( k_Z* BR_hjcc / BR_hjcc_SM,   10/100., 12./100.)  #h->cc channel
           + chisq_fun( k_Z* BR_hjgg / BR_hjgg_SM,   4.5/100., 10./100.)  #h->gg channel
           + chisq_fun( k_Z* BR_hjWW / BR_hjWW_SM,   3.0/100., 4.3/100.)  #h->WW channel
           + chisq_fun( k_Z* BR_hjtautau/ BR_hjtautau_SM,   8.0/100., 5.7/100.)  #h->tautau channel
           + chisq_fun( k_Z* BR_hjZZ / BR_hjZZ_SM,   10/100., 4.3/100.)  #h->ZZ channel
           + chisq_fun( k_Z* BR_hjgaga / BR_hjgaga_SM,   22/100., 5.0/100.) ) #h->gaga channel
  elif ILC:
    # 250GeV Zh production channel
    chisq_mu = ( chisq_fun( k_Z, 0.71/100., 0)  
           + chisq_fun( k_Z* BR_hjbb / BR_hjbb_SM ,  0.46/100., 3.3/100.)  #h->bb channel
           + chisq_fun( k_Z* BR_hjcc / BR_hjcc_SM,   2.9/100., 12./100.)  #h->cc channel
           + chisq_fun( k_Z* BR_hjgg / BR_hjgg_SM,   2.5/100., 10./100.)  #h->gg channel
           + chisq_fun( k_Z* BR_hjWW / BR_hjWW_SM,   1.6/100., 4.3/100.)  #h->WW channel
           + chisq_fun( k_Z* BR_hjtautau/ BR_hjtautau_SM,   1.1/100., 5.7/100.)  #h->tautau channel
           + chisq_fun( k_Z* BR_hjZZ / BR_hjZZ_SM,   6.4/100., 4.3/100.)  #h->ZZ channel
           + chisq_fun( k_Z* BR_hjgaga / BR_hjgaga_SM,   12.0/100., 5.0/100.)  #h->gaga channel
           + chisq_fun( k_Z* BR_hjmumu / BR_hjmumu_SM,   25.5/100., 6.0/100.)  #h->mumu channel
           + chisq_fun( K_W* BR_hjbb / BR_hjbb_SM, 3.7/100., 3.3/100.) ) #(nunu)h->bb channel
    # 350GeV Zh production channel
    chisq_mu += ( chisq_fun( k_Z, 2.0/100., 0)  
           + chisq_fun( k_Z* BR_hjbb / BR_hjbb_SM ,  1.7/100., 3.3/100.)  #h->bb channel
           + chisq_fun( k_Z* BR_hjcc / BR_hjcc_SM,   12.3/100., 12./100.)  #h->cc channel
           + chisq_fun( k_Z* BR_hjgg / BR_hjgg_SM,   9.4/100., 10./100.)  #h->gg channel
           + chisq_fun( k_Z* BR_hjWW / BR_hjWW_SM,   6.3/100., 4.3/100.)  #h->WW channel
           + chisq_fun( k_Z* BR_hjtautau/ BR_hjtautau_SM,   4.5/100., 5.7/100.)  #h->tautau channel
           + chisq_fun( k_Z* BR_hjZZ / BR_hjZZ_SM,   28./100., 4.3/100.)  #h->ZZ channel
           + chisq_fun( k_Z* BR_hjgaga / BR_hjgaga_SM,   43.6/100., 5.0/100.)  #h->gaga channel
           + chisq_fun( k_Z* BR_hjmumu / BR_hjmumu_SM,   97.3/100., 6.0/100.) ) #h->mumu channel
    # 350GeV nunuh production channel
    chisq_mu += (
           + chisq_fun( k_Z* BR_hjbb / BR_hjbb_SM ,  2.0/100., 3.3/100.)  #h->bb channel
           + chisq_fun( k_Z* BR_hjcc / BR_hjcc_SM,   21.2/100., 12./100.)  #h->cc channel
           + chisq_fun( k_Z* BR_hjgg / BR_hjgg_SM,   8.6/100., 10./100.)  #h->gg channel
           + chisq_fun( k_Z* BR_hjWW / BR_hjWW_SM,   6.4/100., 4.3/100.)  #h->WW channel
           + chisq_fun( k_Z* BR_hjtautau/ BR_hjtautau_SM,   17.9/100., 5.7/100.)  #h->tautau channel
           + chisq_fun( k_Z* BR_hjZZ / BR_hjZZ_SM,   22.4/100., 4.3/100.)  #h->ZZ channel
           + chisq_fun( k_Z* BR_hjgaga / BR_hjgaga_SM,   50.3/100., 5.0/100.)  #h->gaga channel
           + chisq_fun( k_Z* BR_hjmumu / BR_hjmumu_SM,   178.9/100., 6.0/100.) ) #h->mumu channel
    # 500GeV Zh production channel
    chisq_mu += ( chisq_fun( k_Z, 1.05/100., 0)  
           + chisq_fun( k_Z* BR_hjbb / BR_hjbb_SM ,  0.63/100., 3.3/100.)  #h->bb channel
           + chisq_fun( k_Z* BR_hjcc / BR_hjcc_SM,   4.5/100., 12./100.)  #h->cc channel
           + chisq_fun( k_Z* BR_hjgg / BR_hjgg_SM,   3.8/100., 10./100.)  #h->gg channel
           + chisq_fun( k_Z* BR_hjWW / BR_hjWW_SM,   1.9/100., 4.3/100.)  #h->WW channel
           + chisq_fun( k_Z* BR_hjtautau/ BR_hjtautau_SM,   1.5/100., 5.7/100.)  #h->tautau channel
           + chisq_fun( k_Z* BR_hjZZ / BR_hjZZ_SM,   8.8/100., 4.3/100.)  #h->ZZ channel
           + chisq_fun( k_Z* BR_hjgaga / BR_hjgaga_SM,   12./100., 5.0/100.)  #h->gaga channel
           + chisq_fun( k_Z* BR_hjmumu / BR_hjmumu_SM,   30./100., 6.0/100.) ) #h->mumu channel
    # 500GeV nunuh production channel
    chisq_mu += (
           + chisq_fun( k_Z* BR_hjbb / BR_hjbb_SM ,  0.23/100., 3.3/100.)  #h->bb channel
           + chisq_fun( k_Z* BR_hjcc / BR_hjcc_SM,   2.2/100., 12./100.)  #h->cc channel
           + chisq_fun( k_Z* BR_hjgg / BR_hjgg_SM,   1.5/100., 10./100.)  #h->gg channel
           + chisq_fun( k_Z* BR_hjWW / BR_hjWW_SM,   0.85/100., 4.3/100.)  #h->WW channel
           + chisq_fun( k_Z* BR_hjtautau/ BR_hjtautau_SM,   2.5/100., 5.7/100.)  #h->tautau channel
           + chisq_fun( k_Z* BR_hjZZ / BR_hjZZ_SM,   3.0/100., 4.3/100.)  #h->ZZ channel
           + chisq_fun( k_Z* BR_hjgaga / BR_hjgaga_SM,   6.8/100., 5.0/100.)  #h->gaga channel
           + chisq_fun( k_Z* BR_hjmumu / BR_hjmumu_SM,   25.0/100., 6.0/100.) ) #h->mumu channel
  return -0.5 * (chisq_mu)



sel_hf = h5py.File('/home/yang/exdisk/CEPC/results/MSSM7/MSSM7_recal_2023.hdf5', 'w')
with h5py.File('/home/yang/exdisk/CEPC/results/MSSM7/MSSM7_1.hdf5', 'r') as f:

    print f.keys()
    group = f["/MSSM7"] 
    sel_group = sel_hf.create_group("/MSSM7")
    
    LogLike = np.array(group["LogLike"])
    CEPC_SM_like = np.array(group["#CEPC_Higgs_SMlike_LogLike @ColliderBit::calc_CEPC_Higgs_SMlike_LogLike"])

 
    Mh_SM_like = np.array(group["#CEPC_Higgs_Couplings @ColliderBit::get_CEPC_Higgs_Couplings::Mh_SM_like"])
    BR_hjcc = np.array(group["#CEPC_Higgs_Couplings @ColliderBit::get_CEPC_Higgs_Couplings::BR_hjcc"])
    BR_hjbb = np.array(group["#CEPC_Higgs_Couplings @ColliderBit::get_CEPC_Higgs_Couplings::BR_hjbb"])
    BR_hjmumu = np.array(group["#CEPC_Higgs_Couplings @ColliderBit::get_CEPC_Higgs_Couplings::BR_hjmumu"])
    BR_hjtautau = np.array(group["#CEPC_Higgs_Couplings @ColliderBit::get_CEPC_Higgs_Couplings::BR_hjtautau"])
    BR_hjWW = np.array(group["#CEPC_Higgs_Couplings @ColliderBit::get_CEPC_Higgs_Couplings::BR_hjWW"])
    BR_hjZZ = np.array(group["#CEPC_Higgs_Couplings @ColliderBit::get_CEPC_Higgs_Couplings::BR_hjZZ"])
    BR_hjgaga = np.array(group["#CEPC_Higgs_Couplings @ColliderBit::get_CEPC_Higgs_Couplings::BR_hjgaga"])
    BR_hjgg = np.array(group["#CEPC_Higgs_Couplings @ColliderBit::get_CEPC_Higgs_Couplings::BR_hjgg"])
    
    CS_Z = np.array(group["#CEPC_Higgs_Couplings @ColliderBit::get_CEPC_Higgs_Couplings::CS_Z"])
    CS_W = np.array(group["#CEPC_Higgs_Couplings @ColliderBit::get_CEPC_Higgs_Couplings::CS_W"])
    
    isvalid = np.array(group["#CEPC_Higgs_Couplings @ColliderBit::get_CEPC_Higgs_Couplings::Mh_SM_like_isvalid"],dtype=np.bool)
    
    CEPC_loglike_new = CEPC_loglike(Mh_SM_like, BR_hjcc, BR_hjbb, BR_hjmumu, BR_hjtautau, BR_hjWW, \
                                      BR_hjZZ, BR_hjgaga, BR_hjgg, CS_Z, CS_W)
    
    print CEPC_loglike_new[isvalid]
    print CEPC_SM_like[isvalid]
    print CEPC_loglike_new[isvalid] - CEPC_SM_like[isvalid]
 
    newLogLike = LogLike - CEPC_SM_like + CEPC_loglike_new
 
    need_list = ["#MSSM7atQ_parameters @MSSM7atQ::primary_parameters::Ad_3",
                 "#MSSM7atQ_parameters @MSSM7atQ::primary_parameters::Au_3",
                 "#MSSM7atQ_parameters @MSSM7atQ::primary_parameters::M2",
                 "#MSSM7atQ_parameters @MSSM7atQ::primary_parameters::Qin",
                 "#MSSM7atQ_parameters @MSSM7atQ::primary_parameters::SignMu",
                 "#MSSM7atQ_parameters @MSSM7atQ::primary_parameters::TanBeta",
                 "#MSSM7atQ_parameters @MSSM7atQ::primary_parameters::mHd2",
                 "#MSSM7atQ_parameters @MSSM7atQ::primary_parameters::mHu2",
                 "#MSSM7atQ_parameters @MSSM7atQ::primary_parameters::mf2",
                 "#LHC_Combined_LogLike @ColliderBit::calc_LHC_LogLike RunI",
                 "#LHC_Combined_LogLike @ColliderBit::calc_LHC_LogLike",
                 "#SL_LL @FlavBit::SL_likelihood buggy",
                 "#b2ll_LL @FlavBit::b2ll_likelihood buggy",
                 "#b2sll_LL @FlavBit::b2sll_likelihood buggy",
                 "#b2sgamma_LL @FlavBit::b2sgamma_likelihood buggy",
                 "#CEPC_Higgs_Couplings @ColliderBit::get_CEPC_Higgs_Couplings::Mh_SM_like_isvalid"
                 ]
 
 
    keys =  group.keys()    
    for key in keys:
      if key == "LogLike":
        print "haha"
        sel_group.create_dataset(key,data= newLogLike )
      elif key == "#CEPC_Higgs_SMlike_LogLike @ColliderBit::calc_CEPC_Higgs_SMlike_LogLike":
        print 2
        sel_group.create_dataset(key,data= CEPC_loglike_new )
      elif key in need_list:
        print key
        sel_group.create_dataset(key,data=np.array(group[key]))


sel_hf.close()

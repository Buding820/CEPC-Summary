#!/usr/bin/env python

import h5py
import numpy as np
import copy
import matplotlib.pyplot as plt

model = "NUHM2"

# Computes the elementwise minimum of the input arrays
def bulk_min(x, limit):
  indices = np.where(x > limit)
  z = copy.deepcopy(x)
  z[indices] = limit
  return z

def chisq_fun(mu, mean, err1, err2):
  return pow(mu-mean,2) / (pow(err1,2)+pow(err2,2))

#BestFit
mh_mean =  125.13615264720315
BR_hjbb_mean =  0.5954769272572568
BR_hjcc_mean =  0.03421785215843392
BR_hjgg_mean =  0.06705868545928646
BR_hjWW_mean =  0.2084732170714905
BR_hjtautau_mean =  0.06379309497425642
BR_hjZZ_mean =  0.026279903281717372
BR_hjgaga_mean =  0.0026957013185202315
BR_hjmumu_mean =  0.00022584909169157805
CS_Z_mean =  0.975177781798079
CS_W_mean =  0.9556326128567615

#  # SM-like
mh_SM = 125.25
BR_hjbb_SM = 0.577
BR_hjcc_SM = 0.0291
BR_hjgg_SM = 0.0857
BR_hjWW_SM = 0.215
BR_hjtautau_SM = 0.0632
BR_hjZZ_SM = 0.0264
BR_hjgaga_SM = 0.00228
BR_hjmumu_SM = 2.19e-04
# SM Higgs normalized cross section
CS_Z_SM = 1.
CS_W_SM = 1.

mu_bb_mean = BR_hjbb_mean/BR_hjbb_SM
mu_cc_mean = BR_hjcc_mean/BR_hjcc_SM
mu_gg_mean = BR_hjgg_mean/BR_hjgg_SM
mu_WW_mean = BR_hjWW_mean/BR_hjWW_SM
mu_tautau_mean = BR_hjtautau_mean/BR_hjtautau_SM
mu_ZZ_mean = BR_hjZZ_mean/BR_hjZZ_SM
mu_gaga_mean = BR_hjgaga_mean/BR_hjgaga_SM
mu_mumu_mean = BR_hjmumu_mean/BR_hjmumu_SM

# theoretical uncertainties = k * SM uncertainties
k = 0.2



def CEPC_loglike(Mh_SM_like, BR_hjcc, BR_hjbb, BR_hjmumu, BR_hjtautau, BR_hjWW, \
                BR_hjZZ, BR_hjgaga, BR_hjgg, CS_Z, CS_W):


#    chisq_mh = chisq_fun(Mh_SM_like, 2., 0.017 , mh_SM)
#    chisq_mu=0.0
  
#  chisq_mu = ( chisq_fun( k_Z, 0.5/100., factor*0)  
#           + chisq_fun( k_Z* BR_hjbb / BR_hjbb_SM ,  0.27/100., factor*3.3/100.)  #h->bb channel
#           + chisq_fun( k_Z* BR_hjcc / BR_hjcc_SM,   3.3/100., factor*12./100.)  #h->cc channel
#           + chisq_fun( k_Z* BR_hjgg / BR_hjgg_SM,   1.3/100., factor*10./100.)  #h->gg channel
#           + chisq_fun( k_Z* BR_hjWW / BR_hjWW_SM,   1.0/100., factor*4.3/100.)  #h->WW channel
#           + chisq_fun( k_Z* BR_hjtautau/ BR_hjtautau_SM,   0.8/100., factor*5.7/100.)  #h->tautau channel
#           + chisq_fun( k_Z* BR_hjZZ / BR_hjZZ_SM,   5.1/100., factor*4.3/100.)  #h->ZZ channel
#           + chisq_fun( k_Z* BR_hjgaga / BR_hjgaga_SM,   6.8/100., factor*5.0/100.)  #h->gaga channel
#           + chisq_fun( k_Z* BR_hjmumu / BR_hjmumu_SM,   17.0/100., factor*6.0/100.)  #h->mumu channel
#           + chisq_fun( CS_W/CS_W_SM* BR_hjbb / BR_hjbb_SM, 2.8/100., factor*3.3/100.) ) #(nunu)h->bb channel



    mu_bb = BR_hjbb/BR_hjbb_SM
    mu_cc = BR_hjcc/BR_hjcc_SM
    mu_gg = BR_hjgg/BR_hjgg_SM
    mu_WW = BR_hjWW/BR_hjWW_SM
    mu_tautau = BR_hjtautau/BR_hjtautau_SM
    mu_ZZ = BR_hjZZ/BR_hjZZ_SM
    mu_gaga = BR_hjgaga/BR_hjgaga_SM
    mu_mumu = BR_hjmumu/BR_hjmumu_SM


    chisq = 0.0

    # 240GeV Zh production channel 20 ab^{-1}
    chisq += ( chisq_fun( CS_Z, CS_Z_mean, 0.26/100., 0)  
           + chisq_fun( CS_Z*mu_bb, CS_Z_mean*mu_bb_mean, 0.14/100., k*3.3/100.)  #h->bb channel
           + chisq_fun( CS_Z*mu_cc, CS_Z_mean*mu_cc_mean,   2.02/100., k*12./100.)  #h->cc channel
           + chisq_fun( CS_Z*mu_gg, CS_Z_mean*mu_gg_mean,   0.81/100., k*10./100.)  #h->gg channel
           + chisq_fun( CS_Z*mu_WW, CS_Z_mean*mu_WW_mean,   0.53/100., k*4.3/100.)  #h->WW channel
           + chisq_fun( CS_Z*mu_tautau, CS_Z_mean*mu_tautau_mean,   0.42/100., k*5.7/100.)  #h->tautau channel
           + chisq_fun( CS_Z*mu_ZZ, CS_Z_mean*mu_ZZ_mean,   4.17/100., k*4.3/100.)  #h->ZZ channel
           + chisq_fun( CS_Z*mu_gaga, CS_Z_mean*mu_gaga_mean,   3.02/100., k*5.0/100.)  #h->gaga channel
           + chisq_fun( CS_Z*mu_mumu, CS_Z_mean*mu_mumu_mean,   6.36/100., k*6.0/100.) ) #h->mumu channel
           
    # 365GeV Zh production channel
    chisq += ( chisq_fun( CS_Z, CS_Z_mean, 1.4/100., 0)  
           + chisq_fun( CS_Z*mu_bb, CS_Z_mean*mu_bb_mean, 0.9/100., k*3.3/100.)  #h->bb channel
           + chisq_fun( CS_Z*mu_cc, CS_Z_mean*mu_cc_mean,   8.8/100., k*12./100.)  #h->cc channel
           + chisq_fun( CS_Z*mu_gg, CS_Z_mean*mu_gg_mean,   3.4/100., k*10./100.)  #h->gg channel
           + chisq_fun( CS_Z*mu_WW, CS_Z_mean*mu_WW_mean,   2.8/100., k*4.3/100.)  #h->WW channel
           + chisq_fun( CS_Z*mu_tautau, CS_Z_mean*mu_tautau_mean,   2.1/100., k*5.7/100.)  #h->tautau channel
           + chisq_fun( CS_Z*mu_ZZ, CS_Z_mean*mu_ZZ_mean,   20./100., k*4.3/100.)  #h->ZZ channel
           + chisq_fun( CS_Z*mu_gaga, CS_Z_mean*mu_gaga_mean,   11./100., k*5.0/100.)  #h->gaga channel
           + chisq_fun( CS_Z*mu_mumu, CS_Z_mean*mu_mumu_mean,   41./100., k*6.0/100.) )#h->mumu channel
    # 365GeV nunuh production channel
    chisq += ( 
             chisq_fun( CS_Z*mu_bb, CS_Z_mean*mu_bb_mean, 1.1/100., k*3.3/100.)  #h->bb channel
           + chisq_fun( CS_Z*mu_cc, CS_Z_mean*mu_cc_mean,   16./100., k*12./100.)  #h->cc channel
           + chisq_fun( CS_Z*mu_gg, CS_Z_mean*mu_gg_mean,   4.5/100., k*10./100.)  #h->gg channel
           + chisq_fun( CS_Z*mu_WW, CS_Z_mean*mu_WW_mean,   4.4/100., k*4.3/100.)  #h->WW channel
           + chisq_fun( CS_Z*mu_tautau, CS_Z_mean*mu_tautau_mean,   4.2/100., k*5.7/100.)  #h->tautau channel
           + chisq_fun( CS_Z*mu_ZZ, CS_Z_mean*mu_ZZ_mean,   21./100., k*4.3/100.)  #h->ZZ channel
           + chisq_fun( CS_Z*mu_gaga, CS_Z_mean*mu_gaga_mean,   16./100., k*5.0/100.)  #h->gaga channel
           + chisq_fun( CS_Z*mu_mumu, CS_Z_mean*mu_mumu_mean,   57./100., k*6.0/100.) )#h->mumu channel
          

    return -0.5 * chisq

for jjj in ["3", "4", "5","6","7","12","13","14"]:
    commen_path = '/home/yang/exdisk/CEPC/results/NUHM2/nuhm2_data/NUHM2_above_1sigma_new_'+jjj
#for jjj in [""]:
#    commen_path = '/home/yang/exdisk/CEPC/results/NUHM2/NUHM2_within_1sigma_new'+jjj
    sel_hf = h5py.File(commen_path+'_recal_2023.hdf5', 'w')
    with h5py.File(commen_path+'.hdf5', 'r') as f:

        print f.keys()
        group = f["/"+model] 
        sel_group = sel_hf.create_group("/"+model)
        
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
        
        print min(CEPC_loglike_new[isvalid])
        print max(CEPC_loglike_new[isvalid])
        
        print CEPC_loglike_new[isvalid]
        print CEPC_SM_like[isvalid]
        print CEPC_loglike_new[isvalid] - CEPC_SM_like[isvalid]
     
        newLogLike = LogLike - CEPC_SM_like + CEPC_loglike_new
     
     
        need_list = [ "#NUHM2_parameters @NUHM2::primary_parameters::A0", "#NUHM2_parameters @NUHM2::primary_parameters::M0", "#NUHM2_parameters @NUHM2::primary_parameters::M12", "#NUHM2_parameters @NUHM2::primary_parameters::SignMu", "#NUHM2_parameters @NUHM2::primary_parameters::TanBeta", "#NUHM2_parameters @NUHM2::primary_parameters::mHu", "#NUHM2_parameters @NUHM2::primary_parameters::mHd", "#LHC_Combined_LogLike @ColliderBit::calc_LHC_LogLike RunI", "#LHC_Combined_LogLike @ColliderBit::calc_LHC_LogLike", "#SL_LL @FlavBit::SL_likelihood buggy", "#b2ll_LL @FlavBit::b2ll_likelihood buggy", "#b2sll_LL @FlavBit::b2sll_likelihood buggy", "#b2sgamma_LL @FlavBit::b2sgamma_likelihood buggy", "LogLike", "LogLike_isvalid", "#CEPC_Higgs_SMlike_LogLike @ColliderBit::calc_CEPC_Higgs_SMlike_LogLike"]
     
     
     
        keys =  group.keys()    
        for key in keys:
          if key == "LogLike":
            print "Yeah!"
            sel_group.create_dataset(key,data= newLogLike )
          elif key == "#CEPC_Higgs_SMlike_LogLike @ColliderBit::calc_CEPC_Higgs_SMlike_LogLike":
            print "Yeah 2!"
            sel_group.create_dataset(key,data= CEPC_loglike_new )
          elif key in need_list:
            sel_group.create_dataset(key,data=np.array(group[key]))


    sel_hf.close()





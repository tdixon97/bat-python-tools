
void filter_mcmc(TString path,TString tree_name="l200a_vancouver_workshop_v0_2_new_m2_mcmc"){

TFile *f1 = new TFile(path+".root");
   TTree *ntuple = (TTree*) f1->Get(tree_name);
   TFile *f2 = new TFile(path+"_small.root","recreate");
   TTree *small = ntuple->CopyTree("Phase!=-1");
}
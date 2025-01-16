
#include <unistd.h>
#include "TString.h"
#include <iostream>
#include "TFile.h"
#include "TTree.h"
#include <TKey.h>
#include <TList.h>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void filter_mcmc(TString path)
{
    std::ifstream file(path);
    json data;
    file >> data;
   TString data_path =TString("../hmixfit/"+(std::string)data["output-dir"]+"/hmixfit-"+(std::string)data["id"]+"/mcmc.root");
   TFile *f1 = new TFile(TString(data_path));
   data_path.Remove(data_path.Length()-5);
   // Get the key name
   TIter nextkey(f1->GetListOfKeys());
   TKey *key;
   bool found=false;
   TString tree_name;

   while ((key = (TKey*)nextkey()) && (found==false))
   {
        // Check if the key name contains the pattern "mcmc"
        if (std::string(key->GetName()).find("mcmc") != std::string::npos)
         {
            tree_name =key->GetName();
         }
   }

   TTree *ntuple = (TTree*) f1->Get(tree_name);
   TFile *f2 = new TFile(data_path+"_small.root","recreate");
   TTree *small = ntuple->CopyTree("Phase!=-1");
   f2->Close();
}

void usage()
{
   std::cout<<"THe usage of this program is:"<<std::endl;
   std::cout<<"./filter_mcmc -f {FILENAME}"<<std::endl;
   std::cout<<"This {FILENAME} is required, \nIt is the path to the mcmc file you wish to remove the prerun from"<<std::endl;

}

int main(int argc, char* argv[]) {

   TString * path=nullptr;
   int opt;

   while ((opt = getopt(argc, argv, "f:h")) != -1) {
        switch (opt) {
            case 'h':
                usage();
                return 1;
            case 'f':
                path = new TString(optarg);
                break;
            default:
               usage();
               return 1;
        }
    }

   if (path!=nullptr)
   {
      filter_mcmc(*path);
   }
   else
   {
      std::cout<<"Error path must be supplied"<<std::endl;
      usage();
      return 0;
   }

}

#include "net.h"
#include <thread>
#include <iostream>


const std::string      trainfile = "train.txt";       //ѵ�����ݵ�ַ
const std::string        prefile = "pre.txt";         //Ԥ�����ݵ�ַ
const std::string       logofile = "logo.txt";        //��־���ݵ�ַ
const std::string       costfile1 = "cost1.txt";        //�ɱ����ݵ�ַ
const std::string       costfile2 = "cost2.txt";        //�ɱ����ݵ�ַ
const std::string       costfile3 = "cost3.txt";        //�ɱ����ݵ�ַ
const std::string       costfile4 = "cost4.txt";        //�ɱ����ݵ�ַ

const double alpha = 0.001;
const int    trainnum = 5000;
std::vector<unsigned long> layernode= {2, 6,16, 1};

void f1() {
	net bpnn;
	bpnn.load_trainset(trainfile);
	bpnn.set_layernum(layernode.size());
	bpnn.set_layernode(layernode);
	bpnn.set_alpha(alpha);
	bpnn.train(trainnum, (train_type)1);
	bpnn.download(costfile1);
	return;
}
void f2() {
	net bpnn;
	bpnn.load_trainset(trainfile);
	bpnn.set_layernum(layernode.size());
	bpnn.set_layernode(layernode);
	bpnn.set_alpha(alpha);
	bpnn.train(trainnum, (train_type)2);
	bpnn.download(costfile2);
	return;
}
void f3() {
	net bpnn;
	bpnn.load_trainset(trainfile);
	bpnn.set_layernum(layernode.size());
	bpnn.set_layernode(layernode);
	bpnn.set_alpha(alpha);
	bpnn.train(trainnum, (train_type)3);
	bpnn.download(costfile3);
	return;
}
void f4() {
	net bpnn;
	bpnn.load_trainset(trainfile);
	bpnn.set_layernum(layernode.size());
	bpnn.set_layernode(layernode);
	bpnn.set_alpha(alpha);
	bpnn.train(trainnum, (train_type)4);
	bpnn.download(costfile4);
	return;
}

int main() {

	std::thread th1(f1);
	std::thread th2(f2);
	std::thread th3(f3);
	std::thread th4(f4);

	th1.join();
	th2.join();
	th3.join();
	th4.join();
	
	return 0;
}

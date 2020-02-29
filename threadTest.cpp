#include <iostream>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>
#include <atomic>
#include <iostream>
#include <vector>
#include <random>
#include <queue>

using namespace std;

//namespace chrono = std::chrono;
using XY = short;

#include <spdlog/spdlog.h>
#include <spdlog/async.h> //support for async logging.
#include <spdlog/sinks/basic_file_sink.h>

#include "board_config.h"
#include "concurrent_queue.h"

auto logger = spdlog::basic_logger_mt<spdlog::async_factory>("go3_logger", "go3_log.txt", true);

struct ProcessData {
	std::mutex mtx_;
	std::condition_variable cond_;

	bool data_ready_ = false;
	std::atomic<int> x{ 3 };

public:
	// 処理に必要なデータの準備をする
	void prepare_data_for_processing()
	{
		// ...準備処理...
		std::chrono::milliseconds dura(1000);
		std::this_thread::sleep_for(dura);    // 2000 ミリ秒

		{
			std::lock_guard<std::mutex> lk(mtx_);
			data_ready_ = true;
			//x = 0;
		}

		// 準備完了したので待機スレッドを全て起床させる
		cond_.notify_all();
	}

	void wait_for_data_to_process1()
	{
		auto start = std::chrono::system_clock::now();      // 計測スタート時刻を保存

		std::unique_lock<std::mutex> lk(mtx_);

		// データの準備ができるまで待機してから処理する
		while (!data_ready_) {
			// 述語を指定しないバージョン
			// 3秒でタイムアウト
			std::cv_status result = cond_.wait_for(lk, chrono::seconds(3));
			if (result == std::cv_status::timeout) {
				std::cout << "wait_for_data_to_process1 : timeout" << std::endl;
				auto end = std::chrono::system_clock::now();       // 計測終了時刻を保存
				auto dur = end - start;        // 要した時間を計算
				auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
				// 要した時間をミリ秒（1/1000秒）に変換して表示
				std::cout << "process1:" << msec << " msec \n";
				return;
			}
		}
		process_data();
		auto end = std::chrono::system_clock::now();       // 計測終了時刻を保存
		auto dur = end - start;        // 要した時間を計算
		auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
		// 要した時間をミリ秒（1/1000秒）に変換して表示
		std::cout << "process1:" << msec << " msec \n";
	}

	void wait_for_data_to_process2()
	{
		auto start = std::chrono::system_clock::now();      // 計測スタート時刻を保存

		std::unique_lock<std::mutex> lk(mtx_);

		// データの準備ができるまで待機してから処理する

		// 述語を指定するバージョン
		// 3秒でタイムアウト
		bool cond = cond_.wait_for(lk, chrono::milliseconds(3000), [this] { return data_ready_; });
		std::cout << "cond = " << cond << std::endl;
		if (!cond) {
			// data_ready == false
			std::cout << "process2 data is not ready:" << std::this_thread::get_id() << std::endl;

			auto end = std::chrono::system_clock::now();       // 計測終了時刻を保存
			auto dur = end - start;        // 要した時間を計算
			auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
			// 要した時間をミリ秒（1/1000秒）に変換して表示
			std::cout << "process2:" << msec << " msec \n";

			return;
		}
		process_data();

		auto end = std::chrono::system_clock::now();       // 計測終了時刻を保存
		auto dur = end - start;        // 要した時間を計算
		auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
		// 要した時間をミリ秒（1/1000秒）に変換して表示
		std::cout << "process2:" << msec << " msec \n";
	}

private:
	void process_data()
	{
		// ...データを処理する...
		std::cout << "process data:" << data_ready_ << ":" << std::this_thread::get_id() << std::endl;
		std::cout << "x:" << x << std::endl;

		x.store(2);
	}
};

const int BOARD_SIZE_MAX = 19;

typedef unsigned long long int64;
int64 zhash_[8][4][EBVCNT];
int64 zhash[8][4][BVCNT];

inline XY get_xy(const int x, const int y)
{
	return x + 20 * y;
}

void initZobristHash()
{
	logger->debug("initZobristhash");

	// axis=0: symmetrical index
	//         [0]->original [1]->90deg rotation, ... [4]->inverting, ...
	// axis=1: stone color index
	//         [0]->blank [1]->black [2]->white [3]->ko
	// axis=2: vertex position

	std::mt19937_64 mt_64_(123);

	for (int i = 0; i < 8; ++i) {
		bool is_inverted = (i == 4);

		for (int j = 0; j < 4; ++j) {
			for (int k = 0; k < EBVCNT; ++k) {

				int x = EBSIZE - 1 - etox[k];
				int y = etoy[k];

				//logger->debug("x={}, y={}", x, y);

				if (x >= 1 && x <= BOARD_SIZE_MAX && y >= 1 && y <= BOARD_SIZE_MAX)
				{
					if (i == 0)
					{
						zhash[i][j][get_xy(x, y)] = zhash_[i][j][k] = mt_64_();
						logger->debug("hash:({:>2},{:>2}) [{}][{}][{}] = {}", x, y, i, j, get_xy(x, y), zhash[i][j][get_xy(x, y)]);
					}
					else if (is_inverted) {
						zhash[i][j][get_xy(x, y)] = zhash_[i][j][k] = zhash_[0][j][xytoe[x][y]];
						logger->debug("hash:({:>2},{:>2}) [{}][{}][{}] = {}", x, y, i, j, get_xy(x, y), zhash[i][j][get_xy(x, y)]);
					}
					else {
						zhash[i][j][get_xy(x, y)] = zhash_[i][j][k] = zhash_[i - 1][j][xytoe[y][x]];
						logger->debug("hash:({:>2},{:>2}) [{}][{}][{}] = {}", x, y, i, j, get_xy(x, y), zhash[i][j][get_xy(x, y)]);
					}
				}
			}
		}
	}
}

// 比較関数
bool my_compare(pair<float, XY> a, pair<float, XY> b) {

	// 基本はfirstで比較
	if (a.first != b.first) {
		// return a.first < b.first; // 昇順
		return a.first > b.first; // 降順
	}

	// それ以外はsecondで比較
	if (a.second != b.second) {
		return a.second < b.second;
	}
	else {
		// どちらも同じ
		return true;
	}
}

void printNormalVector(vector<float>& vect) {
	for (int x : vect) {
		cout << x << endl;
	}
}

void printPairVector(vector<pair<float, XY> >& vect) {
	for (auto x : vect) {
		cout << x.first << " " << x.second << endl;
	}
}

void sort(void) 
{
	vector<pair<float, XY> > vect2;
	vect2.push_back(make_pair(1.01, 3));
	vect2.push_back(make_pair(4.05, 2));
	vect2.push_back(make_pair(2.3, 10));
	vect2.push_back(make_pair(10.09, 1));
	vect2.push_back(make_pair(8.8, 7));

	sort(vect2.begin(), vect2.end());
	printPairVector(vect2);
	cout << "\n" << endl;
	sort(vect2.begin(), vect2.end(), my_compare);
	printPairVector(vect2);
}

void array_trans()
{
	int nums[3][5][7];


	struct uct_node_t
	{
		XY xy;
		int child_num;
	};

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 5;j ++)
		{
			for (int k = 0; k < 7; k++)
			{
				nums[i][j][k] = i * 100 + j * 10 + k;
			}
		}
	}

	//↑と同等
	using array3d = std::array<std::array<std::array<int, 7>, 5>, 3>;
	array3d ar, br;
	vector<pair<int, array3d>> pairs, v;

	std::queue<vector<pair<int, array3d>>> que;

	concurrent_queue<vector < pair<int, array3d>>> c_que;


	for (int i = 0; i < ar.size(); i++)
	{
		for (int j = 0; j < ar[i].size(); j++)
		{
			for (int k = 0; k < ar[i][j].size(); k++)
			{
				ar[i][j][k] = i * 100 + j * 10 + k;
			}

		}
	}

	pairs.push_back(make_pair(1, ar));

	cout << "(1)" << endl;

	c_que.push(pairs);

	cout << "(2)" << endl;

	auto p = c_que.pop();

	cout << "(3)" << endl;

	v = p.get();

	int a = v[0].first;
	br = v[0].second;

	cout << "a=" << a << endl;

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			for (int k = 0; k < 7; k++)
			{
				cout << br[i][j][k] << " " << ar[i][j][k] << endl;
			}
		}
	}
}

#include <stdio.h>
#include <stdlib.h>
#include <regex>

std::string final_score(std::string sgf)
{
	FILE* fp;
	char  buf[1024];
	std::string cmd = R"(C:\Users\tksano\Documents\igo\test\gnugo-mingw-36.exe --score finish --chinese-rule -l)";

	std::string s = cmd + " " + sgf;

	//regex rx(R"(([BW])\+(\d+(?:\.\d+)?))");
	regex rx(R"((Black|White) wins by (\d+(?:\.\d+)?))");
	std::smatch match;
	std::string buf_st;
	std::string col;
	std::string result;

	if ((fp = _popen(s.c_str(), "r")) != NULL) {
		while (fgets(buf, sizeof(buf), fp) != NULL) {
			printf("%s", buf);
			buf_st = buf;

			std::cout << "st:" << buf_st;
			if (regex_search(buf_st, match, rx))
			{
				//for (int j = 0; j < match.size(); j++)
				//{
					cout << "match:" << endl;
					match[1] == "Black" ? col = "B+" : col = "W+";
					result = col.append(match[2]);
				//}
			}
			else 
			{
				cout << "no match" << endl;
			}
		}
		_pclose(fp);
	}

	return result;
}

void reg_test()
{
	string str[3] = { "int", "double[4]", "char[32]" }; // 対象文字列
	string str2 = { "Result from file: W+0.5" };
	//string str2 = { "RBW" };

	std::regex re("(int|double|char)(\\[(\\d+)\\])?"); // 正規表現
	std::regex re2("([BW])\\+(\\d+(?:\\.\\d+)?)");

	std::smatch match; // string型の文字列に対するマッチング結果を格納するのがstd::smatch型
	std::smatch match2; // string型の文字列に対するマッチング結果を格納するのがstd::smatch型

	for (int i = 0; i < 3; ++i) {
		if (regex_match(str[i], match, re)) { // regex_match()は正規表現にマッチすると1を返す
			cout << "matched: ";
			for (int j = 0; j < match.size(); ++j) { // match.size()はサブグループの数+1を返す
				cout << match[j] << ":";
			}
			cout << endl;
		}
	}

	if (regex_search(str2, match2, re2)) { // regex_match()は正規表現にマッチすると1を返す
		cout << "matched: ";
		for (int j = 0; j < match2.size(); ++j) { // match.size()はサブグループの数+1を返す
			cout << match2[j] << ":";
		}
		cout << endl;
	}
	else
	{
		cout << "no matched";
	}

}

#include <iostream>
#include <iomanip>
#include <sstream>

string getDatetimeStr() {
	time_t t = time(nullptr);
	struct tm local;
	
	errno_t err = localtime_s(&local, &t);

	if (err != 0)
	{
		printf("エラー発生\n");
		getchar();
		return 0;
	}

	std::stringstream ss;
	ss << "20" << local.tm_year - 100;
	// setw(),setfill()で0詰め
	ss << setw(2) << setfill('0') << local.tm_mon + 1;
	ss << setw(2) << setfill('0') << local.tm_mday;
	ss << setw(2) << setfill('0') << local.tm_hour;
	ss << setw(2) << setfill('0') << local.tm_min;
	ss << setw(2) << setfill('0') << local.tm_sec;
	// std::stringにして値を返す

	return ss.str();
}

int main()
{
	logger->set_pattern("[%m/%d %H:%M:%S.%e][%t] %v");
	logger->set_level(spdlog::level::debug);
	logger->flush_on(spdlog::level::debug);

	ProcessData p;

	std::thread t1([&] { p.prepare_data_for_processing(); });
	std::thread t2([&] { p.wait_for_data_to_process1(); });
	std::thread t3([&] { p.wait_for_data_to_process2(); });

	t1.join();
	t2.join();
	t3.join();

	//initZobristHash();

	//sort();

	//array_trans();

	std::string s = final_score(R"(C:\Users\tksano\Documents\igo\test\go3_2020-0224-174610-0.sgf")");

	cout << "final_score=" << s << endl;

	cout << getDatetimeStr() << endl;

	reg_test();
}

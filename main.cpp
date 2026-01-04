#include "bits/stdc++.h"
#include "iomanip"

//#include "Moudle.h"

using namespace std;

//诸天神佛，佑我上分
//2022.12.27

#define fastIO() ios::sync_with_stdio(false),cin.tie(nullptr),cout.tie(nullptr)
#define pb push_back
#define judge(x) if (x) cout << "Yes" << endl; else cout << "No" << endl
#define printlist(l, n) cout << "[ "; for(int (i)=0; (i)<(n); ++(i)) {cout << (l)[i] << " "; } cout << "]" << endl;
#define loop(t) while (t--)
#define rep(a, b) for (int i = a; i < b; ++i)
#define rrep(a, b) for (int i = a; i >= b; --i)
#define mem(a, b) memset(a,b,sizeof(a))
#define all(a) a.begin(), a.end()
#define rall(a) rbegin(a), rend(a)
#define put(a) for (auto &_x: a) cin >> _x
#define space() cout << endl

typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> pii;

const int N = 2e5 + 5;
const double eps = 1e-6;
const int maxN = 1000001;
constexpr int mod = 1e9 + 7;
constexpr int MOD = 998244353;
constexpr int i32 = 0x3f3f3f3f;

using vi = std::vector<int>;
using vvi = std::vector<vi>;

class Solution {
public:
    int countPartitions(vector<int>& nums) {
        int ans = 0;

        vector<int> presum(nums.size(), 0);

        for(int i = 1; i <= nums.size(); ++i) {
            presum[i] = presum[i-1] + nums[i-1];
        }
        for(int i = 1; i < nums.size(); ++i) {
            int l = presum[i] - presum[0], r = presum[nums.size() - i];
            if (abs(r - l) % 2 ==0)++ans;
        }

        return ans;
    }
};
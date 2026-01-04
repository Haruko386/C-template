#include "bits/stdc++.h"
#include <cstdlib> // 用于随机数生成
#include <ctime>   // 用于初始化随机数种子

#define all(a) a.begin(), a.end()

using namespace std;

typedef long long ll;

inline int gcd(int a, int b) {
    while (b ^= a ^= b ^= a %= b);
    return a;
}

inline ll lcm(ll a, ll b) {
    return a * b / gcd(a, b);
}

static bool hasTag(long long bit, int index) {
    return ((bit >> index) & 1) != 0;
}

inline int read() {
    int n = 0, f = 1, ch = getchar();
    while (ch < '0' || ch > '9') {
        if (ch == '-')f = -1;
        ch = getchar();
    }
    while (ch >= '0' && ch <= '9') {
        n = n * 10 + ch - '0';
        ch = getchar();
    }
    return n * f;
}

template<typename T>
T lowBit(T x) {
    return x & -x;
}

//快速幂
template<typename T>
T power(T a, ll b) {
    T res = 1;
    for (; b; b /= 2, a *= a)
        if (b % 2)
            res *= a;
    return res;
}

template<typename typC>
bool isPrime(typC num) {
    if (num == 1 || num == 4)return 0;
    if (num == 2 || num == 3)return 1;
    if (num % 6 != 1 && num % 6 != 5)return 0;
    typC tmp = sqrt(num);
    for (int i = 5; i <= tmp; i += 6)
        if (num % i == 0 || num % (i + 2) == 0)return 0;
    return 1;
}

template<typename typC, typename typD>
istream &operator>>(istream &cin, pair<typC, typD> &a) { return cin >> a.first >> a.second; }

template<typename typC>
istream &operator>>(istream &cin, vector<typC> &a) {
    for (auto &x: a) cin >> x;
    return cin;
}

template<typename typC, typename typD>
ostream &operator<<(ostream &cout, const pair<typC, typD> &a) { return cout << a.first << ' ' << a.second; }

template<typename typC, typename typD>
ostream &operator<<(ostream &cout, const vector<pair<typC, typD>> &a) {
    for (auto &x: a) cout << x << '\n';
    return cout;
}

template<typename typC>
ostream &operator<<(ostream &cout, const vector<typC> &a) {
    int n = a.size();
    if (!n) return cout;
    cout << a[0];
    for (int i = 1; i < n; i++) cout << ' ' << a[i];
    return cout;
}

double get_angle(double x1, double y1, double x2, double y2, double x3, double y3) {
    double theta = atan2(x1 - x3, y1 - y3) - atan2(x2 - x3, y2 - y3);
    if (theta > M_PI)
        theta -= 2 * M_PI;
    if (theta < -M_PI)
        theta += 2 * M_PI;

    theta = abs(theta * 180.0 / M_PI);
    return theta;
}

string binary(int x) {
    string s = "";
    while (x) {
        if (x % 2 == 0) s = '0' + s;
        else s = '1' + s;
        x /= 2;
    }
    return s;
}

const int maxn = 100100;
int dfn[maxn], low[maxn];
int tim;
int vis[maxn];
int sd[maxn];
std::stack<int> st;
vector<vector<int>> g;

void tarjan(int cur) {
    dfn[cur] = low[cur] = ++tim;
    vis[cur] = 1;
    st.push(cur);
    for (auto &nex: g[cur]) {
        if (!dfn[nex]) {
            tarjan(nex);
            low[cur] = min(low[cur], low[nex]);
        } else if (vis[nex]) {
            low[cur] = min(low[cur], dfn[nex]);
        }
    }
    if (dfn[cur] == low[cur]) {
        while (!st.empty()) {
            auto pos = st.top();
            st.pop();
            vis[pos] = 0;
            sd[pos] = cur;
            if (pos == cur) break;
        }
    }
}
/*
 * 二维前缀和
 */
class MatrixSum {
private:
    vector<vector<int>> sum;
public:
    MatrixSum(vector<vector<int>> &matrix) {
        int m = matrix.size(), n = matrix[0].size();
        // 注意：如果 matrix[i][j] 范围很大，需要使用 long long
        sum = vector<vector<int>>(m + 1, vector<int>(n + 1));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                sum[i + 1][j + 1] = sum[i + 1][j] + sum[i][j + 1] - sum[i][j] + matrix[i][j];
            }
        }
    }

    int query(int r1, int c1, int r2, int c2) {
        return sum[r2 + 1][c2 + 1] - sum[r2 + 1][c1] - sum[r1][c2 + 1] + sum[r1][c1];
    }
};

template<typename T>
struct FenWick {
    int N;
    vector<T> arr;

    FenWick(int sz) : N(sz), arr(sz + 1, 0) {}

    void update(int pos, T val) {
        for (; pos <= N; pos |= (pos + 1)) {
            arr[pos] += val;
        }
    }

    // 获取 [1, pos] 的和
    T get(int pos) {
        T ret = 0;
        for (; pos > 0; --pos) {
            ret += arr[pos];
            pos &= (pos + 1);
        }
        return ret;
    }

    // 获取 [l, r] 的和
    T query(int l, int r) {
        return get(r) - get(l - 1);
    }
};


/**
 * 链表
 */

struct ListNode {
    int val;
    ListNode *next;

    ListNode() : val(0), next(nullptr) {}

    ListNode(int x) : val(x), next(nullptr) {}

    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

/**
 * 二叉树
 */

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;

    TreeNode() : val(0), left(nullptr), right(nullptr) {}

    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}

    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

/**
 * 并查集
 */

class UnionFind {
    vector<int> root;
    vector<int> rank;
public:
    UnionFind(int size) {
        root.resize(size);
        rank.resize(size);
        for (int i = 0; i < size; ++i) {
            root[i] = rank[i] = i;
        }
    }

    int find(int x) {
        if (x == root[x]) return x;
        return root[x] = find(root[x]);
    }

    void connect(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        if (rootX != rootY) {
            if (rank[rootX] > rank[rootY]) {
                root[rootY] = rootX;
            } else if (rank[rootX] < rank[rootY]) {
                root[rootX] = rootY;
            } else {
                root[rootY] = rootX;
                rank[rootX] += 1;
            }
        }
    }

    bool isConnected(int x, int y) {
        return find(x) == find(y);
    }
};

class Trie {
public:
    int nCnt;
    vector<vector<int>> ch;
    vector<int> f;

    int newNode() {
        ch.emplace_back(26, -1);
        f.push_back(0);
        return nCnt++;
    }

    void add(string &s) {
        int now = 0;
        for (char i: s) {
            f[now]++;
            int c = i - 'a';
            if (ch[now][c] == -1) ch[now][c] = newNode();
            now = ch[now][c];
        }
        f[now]++;
    }

    int query(string &s) {
        int now = 0, ret = 0;
        for (char i: s) {
            if (now > 0) ret += f[now];
            int c = i - 'a';
            now = ch[now][c];
        }
        ret += f[now];
        return ret;
    }
};

/**
* FastIO
*/
template<typename T = int>
inline T fRead() {
    T x = 0, w = 1;
    char c = getchar();
    while (c < '0' || c > '9') {
        if (c == '-') w = -1;
        c = getchar();
    }
    while (c <= '9' && c >= '0') {
        x = (x << 1) + (x << 3) + c - '0';
        c = getchar();
    }
    return w == 1 ? x : -x;
}

template<typename T = int>
inline T cRead() {
    T ret;
    cin >> ret;
    return ret;
}

template<typename T = int>
inline void fWrite(T x) {
    if (x < 0) {
        x = -x;
        putchar('-');
    }
    if (x >= 10) fWrite(x / 10);
    putchar(x % 10 + '0');
}

template<typename T = int>
inline void cWrite(T x) {
    cout << x;
}

/**
 * Least Power of 2 and Greater Power of 2
 * @param val
 * @return
 */

int leastPowerOfTwo(int val) {
    return 32 - __builtin_clz(val - 1);
}

int greaterPowerOfTwo(int val) {
    return 32 - __builtin_clz(val);
}

/**
* AVL树
*/

/**
 * @brief AVL 树
 * @author dianhsu
 * @date 2021/05/25
 * @ref https://zh.wikipedia.org/wiki/AVL树
 * */
#include <bits/stdc++.h>

template<class T>
struct AVLNode {
    T data;
    AVLNode<T> *leftChild;
    AVLNode<T> *rightChild;
    int height;

    AVLNode(T data) : data(data), height(1), leftChild(nullptr), rightChild(nullptr) {}

    ~AVLNode() {
        delete leftChild;
        delete rightChild;
    }
};

template<class T>
class AVL {
public:
    AVL() {
        root = nullptr;
    }

    ~AVL() {
        delete root;
    }

    /**
     * @brief 将结点插入到AVL树中
     * @param val 需要插入的值
     * @note 如果发现这个树中已经有这个值存在了，就不会进行任何操作
     * */
    void insert(T val) {
        _insert(&root, val);
    }

    /**
     * @brief 检查结点是否在AVL树中
     * @param val 需要检查的值
     * */
    bool exist(T val) {
        auto ptr = &root;
        while (*ptr != nullptr) {
            if (val == (*ptr)->data) {
                return true;
            } else if (val < (*ptr)->data) {
                *ptr = (*ptr)->leftChild;
            } else {
                *ptr = (*ptr)->rightChild;
            }
        }
        return false;
    }

    /**
     * @brief 找到值为val的结点
     * @param val 目标值
     * @return 返回值为指向该结点的指针的地址
     */
    AVLNode<T> **find(T val) {
        auto ptr = &root;
        while ((*ptr) != nullptr) {
            if (val == (*ptr)->data) {
                break;
            } else if (val < (*ptr)->data) {
                *ptr = (*ptr)->leftChild;
            } else {
                *ptr = (*ptr)->rightChild;
            }
        }
        return ptr;
    }

    /**
     * @brief 删除结点
     * @note 首先找到结点，然后将结点旋转到叶子结点，然后回溯检查树的平衡性
     * @param val 需要删除的结点的值
     * @note 这个地方需要递归寻找该值的结点，因为需要回溯更新平衡树
     * */
    void remove(T val) {
        _remove(&root, val);
    }


private:
    void _remove(AVLNode<T> **ptr, T val) {
        if (*ptr == nullptr) {
            return;
        }
        if ((*ptr)->data == val) {
            _rotateNodeToLeaf(ptr);
        } else if ((*ptr)->data < val) {
            _remove(&((*ptr)->rightChild), val);
        } else {
            _remove(&((*ptr)->leftChild), val);
        }
        // 完了之后回溯，重新平衡二叉树
        _balance(ptr);
        _updateHeight(*ptr);
    }

    /**
     * @brief 将一个结点旋转到叶子结点
     * @param ptr 将要被旋转至叶子的结点的指针的地址
     * @note 旋转的时候，将当前结点旋转到高度比较小的一边。
     */
    void _rotateNodeToLeaf(AVLNode<T> **ptr) {
        // 当前结点已经是叶子结点了
        if ((*ptr)->leftChild == nullptr and (*ptr)->rightChild == nullptr) {
            *ptr = nullptr;
            return;
        }
        int leftHeight = (*ptr)->leftChild != nullptr ? (*ptr)->leftChild->height : 0;
        int rightHeight = (*ptr)->rightChild != nullptr ? (*ptr)->rightChild->height : 0;
        // 左边高度比较小，左旋
        if (leftHeight <= rightHeight) {
            _leftRotate(ptr);
            _rotateNodeToLeaf(&((*ptr)->leftChild));
        } else {
            // 右旋
            _rightRotate(ptr);
            _rotateNodeToLeaf(&((*ptr)->rightChild));
        }
        _balance(ptr);
        _updateHeight(*ptr);
    }

    /**
     * @brief 插入结点
     *
     * */
    void _insert(AVLNode<T> **ptr, T val) {
        if (*ptr == nullptr) {
            *ptr = new AVLNode<T>(val);
            return;
        }
        if (val < (*ptr)->data) {
            _insert(&((*ptr)->leftChild), val);
        } else if (val > (*ptr)->data) {
            _insert(&((*ptr)->rightChild), val);
        } else {
            // 如果当前平衡二叉树中已经存在这个结点了，不做任何处理
            return;
        }
        _balance(ptr);
        _updateHeight(*ptr);
    }

    /**
     * @brief 平衡结点
     *
     * */
    void _balance(AVLNode<T> **ptr) {
        if (*ptr == nullptr) return;
        int leftHeight = (*ptr)->leftChild != nullptr ? (*ptr)->leftChild->height : 0;
        int rightHeight = (*ptr)->rightChild != nullptr ? (*ptr)->rightChild->height : 0;
        if (abs(leftHeight - rightHeight) <= 1) return;

        if (leftHeight < rightHeight) {
            auto rightElement = (*ptr)->rightChild;
            int rightElementLeftHeight = rightElement->leftChild != nullptr ? rightElement->leftChild->height : 0;
            int rightElementRightHeight = rightElement->rightChild != nullptr ? rightElement->rightChild->height : 0;
            if (rightElementLeftHeight < rightElementRightHeight) {
                // RR
                _leftRotate(ptr);
            } else {
                // RL
                _rightRotate(&((*ptr)->rightChild));
                _leftRotate(ptr);
            }
        } else {
            auto leftElement = (*ptr)->leftChild;
            int leftElementLeftHeight = leftElement->leftChild != nullptr ? leftElement->leftChild->height : 0;
            int leftElementRightHeight = leftElement->rightChild != nullptr ? leftElement->rightChild->height : 0;
            if (leftElementLeftHeight > leftElementRightHeight) {
                // LL
                _rightRotate(ptr);
            } else {
                // LR
                _leftRotate(&((*ptr)->leftChild));
                _rightRotate(ptr);
            }
        }
    }

    /**
     * @brief 右旋
     *
     * */
    void _rightRotate(AVLNode<T> **ptr) {
        auto tmp = (*ptr)->leftChild;
        (*ptr)->leftChild = tmp->rightChild;
        tmp->rightChild = *ptr;
        _updateHeight(tmp);
        _updateHeight(*ptr);
        *ptr = tmp;
    }

    /**
     * @brief 左旋
     * */
    void _leftRotate(AVLNode<T> **ptr) {
        auto tmp = (*ptr)->rightChild;
        (*ptr)->rightChild = tmp->leftChild;
        tmp->leftChild = *ptr;
        _updateHeight(tmp);
        _updateHeight(*ptr);
        *ptr = tmp;
    }

    void _updateHeight(AVLNode<T> *ptr) {
        if (ptr == nullptr) return;
        int leftHeight = ptr->leftChild != nullptr ? ptr->leftChild->height : 0;
        int rightHeight = ptr->rightChild != nullptr ? ptr->rightChild->height : 0;
        ptr->height = std::max(leftHeight, rightHeight) + 1;
    }

    AVLNode<T> *root;
};

template <typename T>
struct LoserTree {
    std::vector<int> losers;               // 存储败者树的数组
    std::vector<std::vector<T>> segments;  // Data segments

    explicit LoserTree(int k) : losers(k, -1), segments(k) {}

    /*
     * 调整败者树，更新败者树状态。
     */
    void adjust(int s) {
        int t = (losers.size() + s) / 2;  // 计算从叶子节点的父节点开始的位置
        int current = s;                  // 当前节点

        while (t > 0) {
            if (losers[t] == -1) {
                // 如果父节点为空，直接将当前节点设置为父节点
                losers[t] = current;
                break;
            } else {
                // 比较当前节点和父节点的值，将较小的节点设置为父节点，如果节点为空，设置为最大值
                T a = segments[current].empty() ? std::numeric_limits<T>::max()
                                                : segments[current].front();
                T b = segments[losers[t]].empty() ? std::numeric_limits<T>::max()
                                                  : segments[losers[t]].front();

                if (a > b) {
                    // 如果当前节点的值大于父节点的值，即当前节点为败者，交换当前节点和父节点
                    std::swap(current, losers[t]);
                }
            }
            t /= 2;  // 继续向上调整
        }
        losers[0] = current;  // 将最终的胜者节点设置为根节点
    }

    /*
     * 多路归并，合并所有数据段
     */
    void multiwayMerge() {
        while (true) {
            int winner = losers[0];
            if (segments[winner].empty()) {
                // 如果胜者节点为空，说明所有节点都已经合并完成
                break;
            }
            std::cout << segments[winner].front() << " ";
            segments[winner].erase(segments[winner].begin());
            adjust(winner);
        }
        std::cout << std::endl;
    }

    /*
     * 打印败者树的初始化状态
     */
    void printTree() const {
        for (size_t i = 0; i < segments.size(); ++i) {
            std::cout << "Segment " << i << ": ";
            for (const auto &elem : segments[i]) {
                std::cout << elem << " ";
            }
            std::cout << "\n";
        }
    }
};

void runTest(const std::vector<std::vector<int>> &data) {
    int k = (int)data.size();
    LoserTree<int> tree(k);  // 构造 K 路败者树

    // 初始化败者树
    for (int i = 0; i < k; ++i) {
        tree.segments[i] = data[i];
    }

    for (int i = k - 1; i >= 0; --i) {
        tree.adjust(i);
    }
    tree.multiwayMerge();
}

/**
* 珂朵莉树
*/

namespace Chtholly {
    struct Node {
        int l, r;
        mutable int v;

        Node(int il, int ir, int iv) : l(il), r(ir), v(iv) {}

        bool operator<(const Node &arg) const {
            return l < arg.l;
        }
    };

    class Tree {
    protected:
        auto split(int pos) {
            if (pos > _sz) return odt.end();
            auto it = --odt.upper_bound(Node{pos, 0, 0});
            if (it->l == pos) return it;
            auto tmp = *it;
            odt.erase(it);
            odt.insert({tmp.l, pos - 1, tmp.v});
            return odt.insert({pos, tmp.r, tmp.v}).first;
        }

    public:
        Tree(int sz, int ini = 1) : _sz(sz), odt({Node{1, sz, ini}}) {}

        virtual void assign(int l, int r, int v) {
            auto itr = split(r + 1), itl = split(l);
            // operations here
            odt.erase(itl, itr);
            odt.insert({l, r, v});
        }

    protected:
        int _sz;
        set<Node> odt;
    };
}
/**
* ST表
*/

template<typename iter, typename BinOp>
class SparseTable {
    using T = typename remove_reference<decltype(*declval<iter>())>::type;
    vector<vector<T>> arr;
    BinOp binOp;
public:
    SparseTable(iter begin, iter end, BinOp binOp) : arr(1), binOp(binOp) {
        int n = distance(begin, end);
        arr.assign(32 - __builtin_clz(n), vector<T>(n));
        arr[0].assign(begin, end);
        for (int i = 1; i < arr.size(); ++i) {
            for (int j = 0; j < n - (1 << i) + 1; ++j) {
                arr[i][j] = binOp(arr[i - 1][j], arr[i - 1][j + (1 << (i - 1))]);
            }
        }
    }

    T query(int lPos, int rPos) {
        int h = floor(log2(rPos - lPos + 1));
        return binOp(arr[h][lPos], arr[h][rPos - (1 << h) + 1]);
    }
};

class randomSelect{
public:
    // 分区函数，使用快速排序的划分方法
    static int partition(vector<int>& arr, int left, int right) {
        int pivot = arr[right];  // 选择最右边的元素作为枢轴
        int i = left - 1;        // i 表示小于 pivot 的部分的边界

        for (int j = left; j < right; ++j) {
            if (arr[j] <= pivot) {
                i++;
                swap(arr[i], arr[j]);
            }
        }
        swap(arr[i + 1], arr[right]);
        return i + 1;
    }

// 随机化分区函数
    static int randomizedPartition(vector<int>& arr, int left, int right) {
        int randomIndex = left + rand() % (right - left + 1);
        swap(arr[randomIndex], arr[right]);  // 随机选择一个枢轴并将其放到最后
        return partition(arr, left, right);
    }

// 随机选择函数，查找第 k 小的元素
    static int randomizedSelect(vector<int>& arr, int left, int right, int k) {
        if (left == right) {
            return arr[left];  // 如果子数组中只有一个元素，直接返回该元素
        }

        // 使用随机化分区
        int pivotIndex = randomizedPartition(arr, left, right);

        // 计算 pivot 在数组中的排名
        int order = pivotIndex - left + 1;

        // 判断 pivot 是第 k 小的元素还是需要在子数组中继续查找
        if (order == k) {
            return arr[pivotIndex];
        } else if (order > k) {
            return randomizedSelect(arr, left, pivotIndex - 1, k);
        } else {
            return randomizedSelect(arr, pivotIndex + 1, right, k - order);
        }
    }
};

/**
* KMP
*/

class KMP {
public:
    /**
     * @brief 统计目标串中有多少个模式串
     * @param target 目标字符串
     * @param pattern 模式字符串
     * */
    static int solve(string &target, string &pattern) {
        int ans = 0;
        int idxTarget = 0, idxPattern = 0;
        vector<int> next(std::move(_prefix(pattern)));
        while (idxTarget < target.length()) {
            while (idxPattern != -1 and pattern[idxPattern] != target[idxTarget]) {
                idxPattern = next[idxPattern];
            }
            ++idxTarget;
            ++idxPattern;
            if (idxPattern >= pattern.length()) {
                ++ans;
                idxPattern = next[idxPattern];
            }
        }
        return ans;
    }

private:
    static vector<int> _prefix(const string &pattern) {
        int i = 0, j = -1;
        vector<int> ret(pattern.length() + 1, -1);
        while (i < pattern.length()) {
            while (j != -1 and pattern[i] != pattern[j]) j = ret[j];
            if (pattern[++i] == pattern[++j]) {
                ret[i] = ret[j];
            } else {
                ret[i] = j;
            }
        }
        return ret;
    }
};

/**
* string hash
*/

class StringHash {
public:
    static unsigned BKDR(const std::string &str) {
        unsigned seed = 131; // 31 131 1313 13131 131313 etc..
        unsigned hash = 0;
        for (auto c: str) {
            hash = hash * seed + c;
        }
        return (hash & 0x7FFFFFFF);
    }

    static unsigned AP(const std::string &str) {
        unsigned hash = 0;
        for (int i = 0; i < str.length(); ++i) {
            if (i & 1) {
                hash ^= (~((hash << 11) ^ str[i] ^ (hash >> 5)));
            } else {
                hash ^= ((hash << 7) ^ str[i] ^ (hash >> 3));
            }
        }
        return (hash & 0x7FFFFFFF);
    }

    static unsigned DJB(const std::string &str) {
        unsigned hash = 5381;
        for (auto c: str) {
            hash += (hash << 5) + c;
        }
        return (hash & 0x7FFFFFFF);
    }

    static unsigned JS(const std::string &str) {
        unsigned hash = 1315423911;
        for (auto c: str) hash ^= ((hash << 5) + c + (hash >> 2));
        return (hash & 0x7FFFFFFF);
    }

    static unsigned SDBM(const std::string &str) {
        unsigned hash = 0;
        for (auto c: str) hash = c + (hash << 6) + (hash << 16) - hash;
        return (hash & 0x7FFFFFFF);
    }

    static unsigned PJW(const std::string &str) {
        auto bits_in_unsigned_int = (unsigned) (sizeof(unsigned) * 8);
        auto three_quarters = (unsigned) (bits_in_unsigned_int * 3 / 4);
        auto one_eighth = (unsigned) (bits_in_unsigned_int / 8);
        unsigned high_bits = (unsigned) (0xFFFFFFFF) << (bits_in_unsigned_int - one_eighth);
        unsigned hash = 0;
        unsigned test = 0;
        for (auto c: str) {
            hash = (hash << one_eighth) + c;
            if ((test = hash & high_bits) != 0) {
                hash = (hash ^ (test >> three_quarters)) & (~high_bits);
            }
        }
        return (hash & 0x7FFFFFFF);
    }

    static unsigned ELF(const std::string &str) {
        unsigned hash = 0, x = 0;
        for (auto c: str) {
            hash = (hash << 4) + c;
            if ((x = hash & 0xF0000000ll) != 0) {
                hash ^= (x >> 24);
                hash &= (~x);
            }
        }
        return (hash & 0x7FFFFFFF);
    }
};

/**
* AC Automaton
*/

namespace Automaton {
    struct ACNode {
        vector<int> nex;
        int fail;
        int cnt;

        ACNode() : nex(26, 0), cnt(0), fail(0) {}
    };

    class AC {
    public:
        AC() : nodes(1) {}

        void insert(const string &arg) {
            int cur = 0;
            for (auto &c: arg) {
                int to = c - 'a';
                if (!nodes[cur].nex[to]) {
                    nodes[cur].nex[to] = (int) nodes.size();
                    nodes.emplace_back();
                }
                cur = nodes[cur].nex[to];
            }
            nodes[cur].cnt++;
        }

        void build() {
            queue<int> Q;
            for (int i = 0; i < 26; ++i) {
                if (nodes[0].nex[i]) {
                    Q.push(nodes[0].nex[i]);
                }
            }
            while (!Q.empty()) {
                int cur = Q.front();
                Q.pop();
                for (int i = 0; i < 26; ++i) {
                    if (nodes[cur].nex[i]) {
                        nodes[nodes[cur].nex[i]].fail = nodes[nodes[cur].fail].nex[i];
                        Q.push(nodes[cur].nex[i]);
                    } else {
                        nodes[cur].nex[i] = nodes[nodes[cur].fail].nex[i];
                    }
                }
            }
        }

        int query(const string &arg) {
            int cur = 0, ans = 0;
            for (auto &c: arg) {
                cur = nodes[cur].nex[c - 'a'];
                for (int j = cur; j and nodes[j].cnt != -1; j = nodes[j].fail) {
                    ans += nodes[j].cnt;
                    nodes[j].cnt = -1;
                }
            }
            return ans;
        }

    private:
        vector<ACNode> nodes;
    };
}

/**
* 后缀数组
*/

class SuffixArray {
private:
    void radixSort(int n, int m, int w, vector<int> &sa, vector<int> &rk, vector<int> &bucket, vector<int> &idx) {
        fill(all(bucket), 0);
        for (int i = 0; i < n; ++i) idx[i] = sa[i];
        for (int i = 0; i < n; ++i) ++bucket[rk[idx[i] + w]];
        for (int i = 1; i < m; ++i) bucket[i] += bucket[i - 1];

        for (int i = n - 1; i >= 0; --i) sa[--bucket[rk[idx[i] + w]]] = idx[i];
        fill(all(bucket), 0);
        for (int i = 0; i < n; ++i) idx[i] = sa[i];
        for (int i = 0; i < n; ++i) ++bucket[rk[idx[i]]];
        for (int i = 1; i < m; ++i) bucket[i] += bucket[i - 1];
        for (int i = n - 1; i >= 0; --i) sa[--bucket[rk[idx[i]]]] = idx[i];
    }

public:
    SuffixArray(const string &s) :
            n(s.length() + 1),
            m(max((int) s.length() + 1, 300)),
            rk(2, vector<int>((s.length() + 1) << 1)),
            bucket(max((int) s.length() + 1, 300)),
            idx(s.length() + 1),
            sa(s.length() + 1),
            ht(s.length()) {

        for (int i = 0; i < n; ++i) ++bucket[rk[0][i] = s[i]];
        for (int i = 1; i < m; ++i) bucket[i] += bucket[i - 1];
        for (int i = n - 1; i >= 0; --i) sa[--bucket[rk[0][i]]] = i;
        int pre = 1;
        int cur = 0;
        for (int w = 1; w < n; w <<= 1) {
            swap(cur, pre);
            radixSort(n, m, w, sa, rk[pre], bucket, idx);
            for (int i = 1; i < n; ++i) {
                if (rk[pre][sa[i]] == rk[pre][sa[i - 1]] and rk[pre][sa[i] + w] == rk[pre][sa[i - 1] + w]) {
                    rk[cur][sa[i]] = rk[cur][sa[i - 1]];
                } else {
                    rk[cur][sa[i]] = rk[cur][sa[i - 1]] + 1;
                }
            }
        }
        for (int i = 0, k = 0; i < n - 1; ++i) {
            if (k) --k;
            while (s[i + k] == s[sa[rk[cur][i] - 1] + k]) ++k;
            ht[rk[cur][i] - 1] = k;
        }
    }

    vector<int> sa;
    vector<int> ht;
private:
    int n, m;
    vector<vector<int>> rk;
    vector<int> bucket, idx;
};

class segmentTree {
    int mod = 0x3f3f3f3f;
    int a[100010];

    struct Segment_Tree {
        ll sum, add, mul;
        int l, r;
    } s[100010 * 4];

    void update(int pos) {
        s[pos].sum = (s[pos << 1].sum + s[pos << 1 | 1].sum) % mod;
        return;
    }

    void pushdown(int pos) { //pushdown的维护
        s[pos << 1].sum = (s[pos << 1].sum * s[pos].mul + s[pos].add * (s[pos << 1].r - s[pos << 1].l + 1)) % mod;
        s[pos << 1 | 1].sum =
                (s[pos << 1 | 1].sum * s[pos].mul + s[pos].add * (s[pos << 1 | 1].r - s[pos << 1 | 1].l + 1)) % mod;

        s[pos << 1].mul = (s[pos << 1].mul * s[pos].mul) % mod;
        s[pos << 1 | 1].mul = (s[pos << 1 | 1].mul * s[pos].mul) % mod;

        s[pos << 1].add = (s[pos << 1].add * s[pos].mul + s[pos].add) % mod;
        s[pos << 1 | 1].add = (s[pos << 1 | 1].add * s[pos].mul + s[pos].add) % mod;

        s[pos].add = 0;
        s[pos].mul = 1;
        return;
    }

    void build_tree(int pos, int l, int r) { //建树
        s[pos].l = l;
        s[pos].r = r;
        s[pos].mul = 1;

        if (l == r) {
            s[pos].sum = a[l] % mod;
            return;
        }

        int mid = (l + r) >> 1;
        build_tree(pos << 1, l, mid);
        build_tree(pos << 1 | 1, mid + 1, r);
        update(pos);
        return;
    }

    void mul(int pos, int x, int y, int k) { //区间乘法
        if (x <= s[pos].l && s[pos].r <= y) {
            s[pos].add = (s[pos].add * k) % mod;
            s[pos].mul = (s[pos].mul * k) % mod;
            s[pos].sum = (s[pos].sum * k) % mod;
            return;
        }

        pushdown(pos);
        int mid = (s[pos].l + s[pos].r) >> 1;
        if (x <= mid) mul(pos << 1, x, y, k);
        if (y > mid) mul(pos << 1 | 1, x, y, k);
        update(pos);
        return;
    }

    void add(int pos, int x, int y, int k) { //区间加法
        if (x <= s[pos].l && s[pos].r <= y) {
            s[pos].add = (s[pos].add + k) % mod;
            s[pos].sum = (s[pos].sum + k * (s[pos].r - s[pos].l + 1)) % mod;
            return;
        }

        pushdown(pos);
        int mid = (s[pos].l + s[pos].r) >> 1;
        if (x <= mid) add(pos << 1, x, y, k);
        if (y > mid) add(pos << 1 | 1, x, y, k);
        update(pos);
        return;
    }

    ll AskRange(int pos, int x, int y) { //区间询问
        if (x <= s[pos].l && s[pos].r <= y) {
            return s[pos].sum;
        }
        pushdown(pos);
        ll val = 0;
        int mid = (s[pos].l + s[pos].r) >> 1;
        if (x <= mid) val = (val + AskRange(pos << 1, x, y)) % mod;
        if (y > mid) val = (val + AskRange(pos << 1 | 1, x, y)) % mod;
        return val;
    }
};

template<typename T>
class BigInt {
private:
    T value;

public:
    BigInt(const T& initialValue = T()) : value(initialValue) {}

    // 大数加法
    BigInt operator+(const BigInt& other) const {
        BigInt result;
        T carry = 0;
        T a = value, b = other.value;
        while (a != 0 || b != 0 || carry != 0) {
            T digitA = a % 10;
            T digitB = b % 10;
            T sum = digitA + digitB + carry;
            carry = sum / 10;
            result.value = std::to_string(sum % 10) + result.value;
            a /= 10;
            b /= 10;
        }
        return result;
    }

    // 大数乘法
    BigInt operator*(const BigInt& other) const {
        BigInt result;
        T a = value, b = other.value;
        int shift = 0;
        while (b != 0) {
            T digitB = b % 10;
            T carry = 0;
            BigInt partialResult;
            for (T i : value) {
                T digitA = i % 10;
                T product = digitA * digitB + carry;
                carry = product / 10;
                partialResult.value = std::to_string(product % 10) + partialResult.value;
                a /= 10;
            }
            if (carry != 0) {
                partialResult.value = std::to_string(carry) + partialResult.value;
            }
            std::string shiftZeros(shift, '0');
            partialResult.value += shiftZeros;
            result = result + partialResult;
            shift++;
            b /= 10;
        }
        return result;
    }

    friend std::ostream& operator<<(std::ostream& os, const BigInt& bigint) {
        os << bigint.value;
        return os;
    }
};

template <typename T>
class Matrix {
private:
    vector<vector<T>> mat;
    int n;

public:
    // 构造函数
    Matrix(int size) : n(size), mat(size, vector<T>(size)) {}

    // 通过二维向量初始化矩阵
    Matrix(const vector<vector<T>>& data) : mat(data), n(data.size()) {}


    int size() const {
        return n;
    }


    vector<T>& operator[](int i) {
        return mat[i];
    }

    const vector<T>& operator[](int i) const {
        return mat[i];
    }

    //矩阵加法
    Matrix<T> add(const Matrix<T>& B) const {
        Matrix<T> C(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                C[i][j] = mat[i][j] + B[i][j];
            }
        }
        return C;
    }

    //矩阵减法
    Matrix<T> subtract(const Matrix<T>& B) const {
        Matrix<T> C(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                C[i][j] = mat[i][j] - B[i][j];
            }
        }
        return C;
    }
    // 矩阵乘法
    Matrix<T> strassenMultiply(const Matrix<T>& B) const {
        if (n == 1) {
            return Matrix<T>{{mat[0][0] * B[0][0]}};
        }

        int newSize = n / 2;
        Matrix<T> A11(newSize), A12(newSize), A21(newSize), A22(newSize);
        Matrix<T> B11(newSize), B12(newSize), B21(newSize), B22(newSize);

        // 分块操作
        for (int i = 0; i < newSize; i++) {
            for (int j = 0; j < newSize; j++) {
                A11[i][j] = mat[i][j];
                A12[i][j] = mat[i][j + newSize];
                A21[i][j] = mat[i + newSize][j];
                A22[i][j] = mat[i + newSize][j + newSize];

                B11[i][j] = B[i][j];
                B12[i][j] = B[i][j + newSize];
                B21[i][j] = B[i + newSize][j];
                B22[i][j] = B[i + newSize][j + newSize];
            }
        }

        // 计算 7 个中间结果矩阵
        Matrix<T> M1 = A11.add(A22).strassenMultiply(B11.add(B22));
        Matrix<T> M2 = A21.add(A22).strassenMultiply(B11);
        Matrix<T> M3 = A11.strassenMultiply(B12.subtract(B22));
        Matrix<T> M4 = A22.strassenMultiply(B21.subtract(B11));
        Matrix<T> M5 = A11.add(A12).strassenMultiply(B22);
        Matrix<T> M6 = A21.subtract(A11).strassenMultiply(B11.add(B12));
        Matrix<T> M7 = A12.subtract(A22).strassenMultiply(B21.add(B22));

        // 组合结果矩阵
        Matrix<T> C(n);
        for (int i = 0; i < newSize; i++) {
            for (int j = 0; j < newSize; j++) {
                C[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
                C[i][j + newSize] = M3[i][j] + M5[i][j];
                C[i + newSize][j] = M2[i][j] + M4[i][j];
                C[i + newSize][j + newSize] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
            }
        }

        return C;
    }

    void print() const {
        for (const auto& row : mat) {
            for (const auto& elem : row) {
                cout << elem << " ";
            }
            cout << endl;
        }
    }
};
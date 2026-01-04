#include "bits/stdc++.h"

#define all(a) a.begin(), a.end()

using namespace std;

typedef long long ll;

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
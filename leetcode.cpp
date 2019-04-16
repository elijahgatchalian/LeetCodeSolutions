//
//  Eli Gatchalian
//  LeetCodeSolutions
//
//  Copyright © 2019 Eli Gatchalian. All rights reserved.
//
//
//   Hi everyone! I'm a software engineer and I decided to compile my solutions to
//   some of the problems posted on leetcode.com using C++. I have provided time
//   and space complexity for each problem. Feel free to message me if you see something
//   confusing or wrong!
//

//  Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
    
};

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    
};

#include <vector>
#include <unordered_map>
#include <queue>
#include <iostream>
using namespace std;

int main(int argc, const char * argv[]) {
    // insert code here...
    cout << "Hello, World!\n";
    return 0;
}

//
//  1. Two Sum - Easy
//
//  Given an array of integers, return indices of the two numbers such that
//  they add up to a specific target.
//
//  You may assume that each input would have exactly one solution, and
//  you may not use the same element twice
//
//  Big(O) -> O(n), where n = size of the array
//  Memory -> O(n), where n = size of the array
//

vector<int> twoSum(vector<int>& nums, int target){
    unsigned long size = nums.size();
    unordered_map<int,int> valueToIndex;
    
    for(int i = 0; i < size; i++){
        auto it = valueToIndex.find(target - nums[i]);
        if(it == valueToIndex.end()){
            valueToIndex[nums[i]] = i;
        }else{
            return{it->second, i};
        }
    }
    return {};
}

//
//  2. Add Two Numbers - Medium
//
//  You are given non-empty linked lists representing two non-negative
//  integers. The digits are stored in reverse order and each of their nodes
//  contain a single digit. Add the two numbers and return it as a linked list.
//
//  You may assume the two numbers do not contain any leading zero,
//  except the number 0 itself.
//
//  Big(O) -> O(n), where n = length of l1 and l2
//  Memory -> O(n), where n = length of l1 and l2
//

ListNode* addTwoNumbers(ListNode* l1, ListNode* l2){
    ListNode* dummyHead = new ListNode(0);
    ListNode *p1 = l1, *p2 = l2, *results = dummyHead;
    int carry = 0;
    int x, y, sum;
    while(p1 || p2){
        if(p1) x = p1->val;
        else x = 0;
        
        if(p2) y = p2->val;
        else y = 0;
        
        sum = carry + x + y;
        carry = sum / 10;
        results->next = new ListNode(sum % 10);
        results = results->next;
        
        if(p1) p1 = p1->next;
        if(p2) p2 = p2->next;
    }
    if(carry > 0){
        results->next = new ListNode(carry);
    }
    return dummyHead->next;
}

//
//  3. Longest Substring Without Repeating Characters - Medium
//
//  Given a string, find the length of the longest substring without
//  repeating characters.
//
//  Big(O) -> O(n), where n = length of the string
//  Memory -> O(n), where n = length of the string
//

int lengthOfLongestSubstring(string s) {
    unordered_map<char, int> charToOccurence;
    string currentRun = "";
    int maxRun = 0, index = 0;;
    
    while (index < s.length()){
        auto it = charToOccurence.find(s[index]);
        if(it != charToOccurence.end()){
            index = it->second + 1;
            charToOccurence.clear();
            currentRun = s[index];
        }else{
            currentRun += s[index];
        }
        if ((int)currentRun.length() > maxRun){
            maxRun = (int)currentRun.length();
        }
        charToOccurence[s[index]] = index;
        index++;
    }
    
    return maxRun;
}

//
//  4. Median of Two Sorted Arrays - Hard
//
//  There are two sorted arrays nums1 and nums2 of size m and n
//  respectively.
//
//  Find the median of the two sorted arrays. The overrall run time
//  complexity should be O(log(m+n)).
//
//  You may assume nums1 and nums2 cannot be both empty.
//
//  Big(O) -> O(m+n), where m and n are the sizes of nums1 and nums2 respectively
//  Memory -> O(m+n), where m and n are the sizes of nums1 and nums2 respectively
//

void createPriorityQueue4(const vector<int> nums, priority_queue<int,vector<int>,greater<int>> &min_heap){
    for(int i = 0; i < nums.size(); i++){
        min_heap.push(nums[i]);
    }
}

double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
    priority_queue<int,vector<int>,greater<int>> min_heap;
    int size = (int)nums1.size() + (int)nums2.size();
    
    createPriorityQueue4(nums1, min_heap);
    createPriorityQueue4(nums2, min_heap);
    
    while(min_heap.size() != size/2 + 1){
        min_heap.pop();
    }
    
    double val = min_heap.top();
    
    if(size%2 == 0){
        min_heap.pop();
        val += min_heap.top();
        val /= 2;
    }
    
    return val;
}

//
//  6. ZigZag Conversion - Medium
//  The string "PAYPALISHIRING" is written in a zigzag patter on a given number
//  of rows like this: (you may want to display this pattern in a fixed font for legibility)
//
//  P       A       H       N
//  A   P   L   S   I   I   G
//  Y       I       R
//
//  And then read the line by line: "PAHNAPLSIIGYIR"
//
//  Big(O) -> O(n), where n = size of the string
//  Memory -> O(n), where n = size of the string
//

void switchDirection(const int numRows, int &rowCount, bool &up){
    if(rowCount < 0){
        up = true;
        rowCount = 1;
    }else if(rowCount >= numRows){
        up = false;
        rowCount = numRows - 2;
    }
}

void createVector6(vector<string> &letters, const int numRows, const string s){
    int index = 0, rowCount = 0;
    bool up = true;
    while(index < s.length()){
        letters[rowCount] += s[index];
        
        if(up) rowCount++;
        else rowCount--;
        
        switchDirection(numRows, rowCount, up);
        
        index++;
    }
}

string convert(string s, int numRows) {
    if(numRows < 2) return s;
    
    vector<string> letters(numRows);
    createVector6(letters, numRows, s);
    
    string zigZag = "";
    for(int i = 0; i < numRows; i++){
        zigZag += letters[i];
    }
    
    return zigZag;
}

//
//  7. Reverse Integer - Easy
//
//  Given a 32-bit signed integer, reverse digits of an integer.
//
//  Note:
//  Assume we are dealing with an environment which could only store integers
//  within the 32-bit signed integer range: [-2^31, 2^31 - 1]. For the purpose of this
//  problem, assume that your function returns 0 when the reverse integer overflows.
//
//  Big(O) -> O(log(x))
//  Memory -> O(1)
//

int reverse(int x) {
    if(x == 0 || x == INT_MIN || x == INT_MAX) return 0;
    
    int rev = 0;
    while(x != 0){
        int pop = x % 10;
        x/=10;
        if(rev > INT_MAX/10 || (rev == INT_MAX/10 && pop > 7)){
            return 0;
        }else if(rev < INT_MIN/10 || (rev == INT_MIN/10 && pop < -8)){
            return 0;
        }
        rev = rev * 10 + pop;
    }
    return rev;
}

//
//  11. Container With Most Water - Medium
//
//  Given n non-negative integers  a1, a2, ..., an, where each represents a point at
//  coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i
//  is at (i, ai) and (i, 0). Find the two lines, which together with x-axis forms a
//  container, such that the container contains the most water.
//
//  Note: You may not slant the container and n is at least 2.
//
//  Big(O) -> O(n), where n = size of the array
//  Memory -> O(1)
//
//  ********* See StefanPochmann's post for a thorough explanation *********
//

int maxArea(vector<int>& height) {
    int amountOfWater = 0;
    int left = 0, right = (int)height.size() - 1;
    while(left < right){
        int h = min(height[left], height[right]);
        amountOfWater = max(amountOfWater, (right-left)*h);
        while(height[left] <= h && left < right) left++;
        while(height[right] <= h && left < right) right--;
    }
    return amountOfWater;
}

//
//  41. First Missing Positive - Hard
//
//  Given an unsorted integer array, find the smallest missing positive integer.
//
//  Note:
//  Your algorithm should run in O(n) time and uses constant extra space.
//
//  Big(O) -> O(n), where n = size of the array
//  Memory -> O(1)
//

void moveValues(vector<int> &arr, const int size){
    for(int i = 0; i < size; i++){
        while(arr[i] > 0 && arr[i] < size && arr[arr[i] - 1] != arr[i]){
            swap(arr[i], arr[arr[i] - 1]);
        }
    }
}

int findNumber(const vector<int> arr, const int size){
    for(int i = 0; i < size; i++){
        if(arr[i] != i + 1){
            return i + 1;
        }
    }
    return size + 1;
}

int firstMissingPositive(vector<int>& nums) {
    int size = (int)nums.size();
    moveValues(nums, size);
    return findNumber(nums, size);
}

//
//  70. Climbing Stairs - Easy
//
//  You are climbing a stair case. It takes n steps to reach to the top.
//
//  Each time you can either climb 1 or 2 steps. In how many distinctive ways can you
//  climb to the top?
//
//  Note: Given n will be a positive integer.
//
//  Big(O) -> O(n)
//  Memory -> O(1)
//

int climbStairs(int n) {
    if(n < 3) return n;
    
    int first = 1, second = 2, differentWays = 0;
    
    for(int i = 2; i < n; i++){
        differentWays = first + second;
        first = second;
        second = differentWays;
    }
    
    return differentWays;
}

//
//  75. Sort Colors - Medium
//
//  Given an array with n objects colored red, white or blue, sort them in-place so that objects
//  of the same color are adjacents, with the colors in the order red, white and blue.
//
//  Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue
//  respectively.
//
//  Note: You are not suppose to use the library's sort function for this problem.
//
//  Big(O) -> O(n), where n = size of the array
//  Memory -> O(1)
//

void sortColors(vector<int>& nums) {
    int red = 0, white = 0, blue = 0;
    
    for(int i = 0; i < nums.size(); i++){
        if(nums[i] == 0) red++;
        else if(nums[i] == 1) white++;
        else blue++;
    }
    
    for(int i = 0; i < nums.size(); i++){
        if(i < red) nums[i] = 0;
        else if(i < red + white) nums[i] = 1;
        else nums[i] = 2;
    }
}

//
//  118. Pascal's Triangle - Easy
//
//  Given a non-negative integer numRows, generate the first numRows of Pascal's Triangle.
//
//  Big(O) -> O(n^2), where n = numRows
//  Memory -> O(n^2), where n = numRows
//

vector<vector<int>> generate(int numRows){
    if(numRows <= 0) return {};
    vector<vector<int>> pascalsTriangle = {{1}};
    
    for(int i = 1; i < numRows; i++){
        pascalsTriangle.push_back({1});
        for(int j = 1; j < i; j++){
            int sum = pascalsTriangle[i - 1][j] + pascalsTriangle[i - 1][j - 1];
            pascalsTriangle[i].push_back(sum);
        }
        pascalsTriangle[i].push_back(1);
    }
    
    return pascalsTriangle;
}

//
//  119. Pascal's Triangle II - Easy
//
//  Given a non-negative index k where k ≤ 33, return the kth index row of the Pascal's triangle.
//
//  Note that the row index starts from 0.
//
//  Big(O) -> O(n), where n = rowIndex
//  Memory -> O(n), where n = rowIndex
//

vector<int> getRow(int rowIndex) {
    vector<int> pascalsTriangle(rowIndex + 1, 1); // set entire row to 1
    long value = 1;
    for (int i = 1; i < rowIndex; i++) {
        value = value * (rowIndex - i + 1) / i;
        pascalsTriangle[i] = (int)value;
    }
    return pascalsTriangle;
}

//
//  169. Majority Element - Easy
//
//  Given an array of size n, find the majority element. The majority element is the
//  element that appears more than n/2 times.
//
//  You may assume that the array is non-empty and the majority element alaways
//  exist in the array.
//
//  Big(O) -> O(n), where n = size of the array
//  Memory -> O(n), where n = size of the array
//

int majorityElement(vector<int>& nums){
    int n = (int)nums.size();
    unordered_map<int,int> numToFrequency;
    for(int i = 0; i < n; i++){
        numToFrequency[nums[i]]++;
        if(numToFrequency[nums[i]] > n/2) return nums[i];
    }
    return -1; // This should never be reached since the majority element will exist
}

//
//  200. Number of Islands - Medium
//
//  Given a 2d grid of '1's (land) and '0's (water), count the number of islands. An
//  island is surrounded by water and is formed by connecting adjacent lands
//  horizontally or vertically. You may assume all four edges of the grid are all
//  surrounded by water.
//
//  Big(O) -> O(n x m), where n and m are the dimensions of the 2d array
//  Memory -> O(1)
//
//  ********* This solution originially passed all cases. When I retested it, it kept
//  failing on one test case. To fix this issue, replace the 'valid' function call in
//  'findSurroundingLand' with the criteria check used in 'valid'. I'm not sure why this
//  caused the one test case to fail as the call to 'valid' should be constant. *********
//

bool valid(const vector<vector<char>> grid, const int i, const int j){
    return i >= 0 && i < grid.size() && j >= 0 && j < grid[i].size() && grid[i][j] == '1';
}

void findSurroundingLand(vector<vector<char>> &grid, const int i, const int j){
    if(!valid(grid, i, j)) return;
    
    grid[i][j] = '0'; // make sure not to visit this spot again
    
    findSurroundingLand(grid, i, j + 1); // down
    findSurroundingLand(grid, i, j - 1); // up
    findSurroundingLand(grid, i - 1, j); // left
    findSurroundingLand(grid, i + 1, j); // right
}

int numIslands(vector<vector<char>>& grid) {
    int islandsFound = 0;
    for(int i = 0; i < grid.size(); i++){
        for(int j = 0; j < grid[0].size(); j++){
            if(grid[i][j] == '1'){
                findSurroundingLand(grid, i, j);
                islandsFound++;
            }
        }
    }
    return islandsFound;
}

//
//  215. Kth Largest Element in an Array - Medium
//
//  Find the kth largest element in an unsorted array. Note that it is the kth
//  largest element in the sorted order, not the kth distinct element.
//
//  Note:
//  You may assume k is always valid, 1 ≤ k ≤ array's length.
//
//  Big(O) -> O(nlogn), where n = size of the array
//  Memory -> O(k), where k = kth element. The heap should never be larger than this number.
//

int findKthLargest(vector<int>& nums, int k) {
    priority_queue<int, vector<int>, greater<int>> min_heap;
    for(int i = 0; i < (int)nums.size(); i++){
        min_heap.push(nums[i]);
        if(min_heap.size() > k) min_heap.pop();
    }
    return min_heap.top();
}

//
//  217. Contains Duplicate
//
//  Given an array of integers, find if the array contains any duplicates.
//
//  Your function should return true if any value appears at least twice in the array, and it should
//  return false if every element is distinct.
//
//  Big(O) -> O(n), where n = size of the array.
//  Memory -> O(n), where n = size of the array.
//

bool containsDuplicate(vector<int>& nums) {
    unordered_map<int, int> numbersToOccurence;
    
    for(int i = 0; i < nums.size(); i++){
        numbersToOccurence[nums[i]]++;
        if(numbersToOccurence[nums[i]] > 1) return true;
    }
    
    return false;
}

//
//  230. Kth Smallest Element in a BST - Medium
//
//  Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.
//
//  Note:
//  You may assume k is always valid, 1 ≤ k ≤ BST's total elements.
//
//  Big(O) -> O(nlogn), where n = size of the tree.
//  Memory -> O(k), where k = kth element. The heap should never be larger than this number.
//

void pushValuesToHeap(TreeNode* root, priority_queue<int> &max_heap, const int k){
    if(root == nullptr) return;
    max_heap.push(root->val);
    if(max_heap.size() > k) max_heap.pop();
    pushValuesToHeap(root->left, max_heap, k);
    // since this is a binary tree, only look to the right if the size of max_heap ≤ k
    if(max_heap.size() <= k) pushValuesToHeap(root->right, max_heap, k);
}

int kthSmallest(TreeNode* root, int k) {
    priority_queue<int> max_heap;
    
    pushValuesToHeap(root, max_heap, k);
    
    return max_heap.top();
}

//
//  240. Search a 2D Matrix II - Medium
//
//  Write an efficent algorithm that searches for a value in an m x n matrix. This matrix has
//  the following properties:
//  - Integers in each row are sorted in ascending from left to right.
//  - Integers in each column are sorted in ascending from top to bottom.
//
//  Big(O) -> O(m + n), where m and n are the dimensions of the 2D matrix
//  Memory -> O(1)
//

bool searchMatrix(vector<vector<int>>& matrix, int target) {
    if(matrix.size() == 0){
        return false;
    }
    int x = 0, y = (int)matrix[0].size() - 1;
    while(x < matrix.size() && y >= 0){
        if(matrix[x][y] == target) return true;
        else if(matrix[x][y] < target) x++;
        else y--;
    }
    return false;
}

//
//  307. Range Sum Query - Mutable - Medium
//
//  Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j),
//  inclusive.
//
//  The update(i, val) function modifies nums by updating the element at index i to val.
//
//  NumArray Big(O) -> O(n), where n = size of the array.
//  update Big(O) -> O(n), where n = size of the array.
//  sumRange Big(O) -> O(1)
//  Memory -> O(n), where n = size of the array.
//

class NumArray {
    vector<int> allSums, myArray;
public:
    NumArray(vector<int>& nums) {
        myArray = nums;
        int sum = 0;
        for(int i = 0; i < myArray.size(); i++){
            sum += myArray[i];
            allSums.push_back(sum);
        }
    }
    
    void update(int i, int val) {
        for(int index = i; index < allSums.size(); index++){
            allSums[index] -= myArray[i];
            allSums[index] += val;
        }
        myArray[i] = val;
    }
    
    int sumRange(int i, int j) {
        if(i == 0) return allSums[j];
        else return allSums[j] - allSums[i - 1];
    }
};

//
//  344. Reverse String - Easy
//
//  Write a function that reverses a string. The input string is given as an array
//  of characters char[].
//
//  Do not allocate extra space for another array, you must do this by modifying the
//  input array in-place with O(1) extra memory.
//
//  You may assume all the characters consist of printable ascii characters.
//
//  Big(O) -> O(n), where n = size of the array
//  Memory -> O(1)
//

void reverseString(vector<char>& s) {
    int beginning = 0, ending = (int)s.size() - 1;
    while(beginning < ending){
        swap(s[beginning], s[ending]);
        beginning++;
        ending--;
    }
}

//
//  378. Kth Smallest Element in a Sorted Matrix - Medium
//
//  Given a n x n matrix where each of the rows and columns are sorted in ascending order, find
//  the kth smallest element in the matrix.
//
//  Note that it is the kth smallest element in the sorted order, not the kth distinct element.
//
//  Big(O) -> O(nlogn), where n = size of the matrix.
//  Memory -> O(k), where k = kth element. The heap should never be larger than this number.
//

int kthSmallest(vector<vector<int>>& matrix, int k) {
    priority_queue<int> max_heap;
    for(int i = (int)matrix.size() - 1; i >= 0; i--){
        for(int j = (int)matrix[0].size() - 1; j >= 0; j--){
            if(max_heap.size() < k || matrix[i][j] <= max_heap.top()){
                max_heap.push(matrix[i][j]);
                if(max_heap.size() > k) max_heap.pop();
            }
        }
    }
    return max_heap.top();
}

//
//  509. Fibonacci Number - Easy
//
//  The Fibonacci numbers, commonly denoted F(n) form a sequence, called the Fibonacci sequence,
//  such that each number is the sum of the two preceding numbers, starting from 0 and 1. That is,
//
//  F(0) = 0, F(1) = 1
//  F(N) = F(N - 1) + F(N - 1), for N > 1
//
//  Big(O) -> O(n), where n = N input.
//  Memory -> O(1)
//

int fib(int N) {
    if(N == 0) return 0;
    
    int twoBefore = 0, oneBefore = 1;
    int current = 1;
    
    for(int i = 1; i < N; i++){
        current = oneBefore + twoBefore;
        twoBefore = oneBefore;
        oneBefore = current;
    }
    
    return current;
}

//
//  695. Max Area of Island
//
//  Given a non-empty 2D array grid of 0's and 1's, an island is a group of 1's (representing land)
//  connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid
//  are surrounded by water.
//
//  Find the maximum area of an island in the given 2D array. (If there is no island, the maximum
//  area is 0.)
//
//  Big(O) -> O(n x m), where n and m are the dimensions of the grid.
//  Memory -> O(1)
//
//  ********* This solution originially passed all cases. When I retested it, it kept
//  failing on one test case. To fix this issue, replace the 'valid' function call in
//  'countArea' with the criteria check used in 'valid'. I'm not sure why this
//  caused the one test case to fail as the call to 'valid' should be constant. *********
//

bool valid(const vector<vector<int>> grid, const int x, const int y){
    return x >= 0 && x < grid.size() && y >= 0 && y < grid[0].size() && grid[x][y] == 1;
}

int countArea(vector<vector<int>> &grid, int x, int y){
    if(!valid(grid, x, y)) return 0;
    
    grid[x][y] = 0;
    
    int count = 1;
    count += countArea(grid, x + 1, y);
    count += countArea(grid, x - 1, y);
    count += countArea(grid, x, y + 1);
    count += countArea(grid, x, y - 1);
    
    return count;
}

int maxAreaOfIsland(vector<vector<int>>& grid) {
    int maxArea = 0;
    for(int i = 0; i < grid.size(); i++){
        for(int j = 0; j < grid[0].size(); j++){
            if(grid[i][j] == 1){
                int count = countArea(grid, i, j);
                if(count > maxArea){
                    maxArea = count;
                }
            }
        }
    }
    return maxArea;
}

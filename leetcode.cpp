/*
 Eli Gatchalian
 LeetCodeSolutions

 Copyright © 2019 Eli Gatchalian. All rights reserved.

 Hi everyone! I'm a software engineer and I decided to compile my solutions to
 some of the problems posted on leetcode.com using C++. I have provided time
 and space complexity for each problem. Feel free to message me if you see
 something confusing or wrong!
*/

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

/*
 1. Two Sum - Easy

 Given an array of integers, return indices of the two numbers such that
 they add up to a specific target.

 You may assume that each input would have exactly one solution, and
 you may not use the same element twice

 Big(O) -> O(n), where n = size of the array
 Memory -> O(n), where n = size of the array
*/

vector<int> twoSum(vector<int>& nums, int target){
    /*
     Purpose: Find the indices of the two numbers that add up to the target
     Input:
        - nums: A vector of integers
        - target: An integer representing the target to find
     Output: An integer vector. This vector has the indices of the two numbers
             that add up to the input target.
    */
    int size = (int)nums.size();
    unordered_map<int,int> valueToIndex;
    
    for(int i = 0; i < size; i++){
        auto it = valueToIndex.find(target - nums[i]);
        if(it == valueToIndex.end()) valueToIndex[nums[i]] = i;
        else return {it->second, i};
    }
    
    return {};
}

/*
 2. Add Two Numbers - Medium

 You are given non-empty linked lists representing two non-negative
 integers. The digits are stored in reverse order and each of their nodes
 contain a single digit. Add the two numbers and return it as a linked list.

 You may assume the two numbers do not contain any leading zero,
 except the number 0 itself.

 Big(O) -> O(n), where n = length of l1 and l2
 Memory -> O(n), where n = length of l1 and l2
*/

ListNode* addTwoNumbers(ListNode* l1, ListNode* l2){
    /*
     Purpose: Add the two numbers represented by two singly linked lists
     Input:
        - l1: A linked list of integers
        - l2: A linked list of integers
     Output: A linked list representing the summation of both inputs
    */
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
    
    if(carry > 0) results->next = new ListNode(carry);
    
    return dummyHead->next;
}

/*
 3. Longest Substring Without Repeating Characters - Medium

 Given a string, find the length of the longest substring without
 repeating characters.

 Big(O) -> O(n), where n = length of the string
 Memory -> O(n), where n = length of the string
*/

int lengthOfLongestSubstring(string s) {
    /*
     Purpose: Find the length of the longest substring without
              repeating characters
     Input: A string
     Output: An integer representing the length of the longest run in the input
             string that doesn't repeat characters
    */
    if(s.length() <= 1) return (int)s.length();
    
    // Holds most recent (index + 1) of char
    unordered_map<char, int> charToRecentIndex;
    int maxRun = 0;
    int slow = 0, fast = 0;
    
    while(fast < s.length()){
        //  Update slow to most recent index of the repeated character that
        //      fast just landed on
        if(charToRecentIndex[s[fast]] > slow) slow = charToRecentIndex[s[fast]];
        
        charToRecentIndex[s[fast]] = fast + 1;
        
        fast++;
        maxRun = max(maxRun, fast - slow);
    }
    
    return maxRun;
}

/*
 4. Median of Two Sorted Arrays - Hard

 There are two sorted arrays nums1 and nums2 of size m and n respectively.

 Find the median of the two sorted arrays. The overrall run time
 complexity should be O(log(m+n)).

 You may assume nums1 and nums2 cannot be both empty.

 Big(O) -> O(m+n), where m and n are the sizes of nums1 and nums2 respectively
 Memory -> O(m+n), where m and n are the sizes of nums1 and nums2 respectively
*/

void createPriorityQueue4(const vector<int> nums, priority_queue<int,vector<int>,greater<int>> &min_heap){
    /*
     Purpose: Create the min_heap based on values in the input vector
     Input:
        - nums: A vector of integers
        - min_heap: An empty integer minimum heap
     Output: No output but this function pushes values from num to min_heap.
             min_heap is changed via pass-by-reference.
    */
    for(int i = 0; i < nums.size(); i++){
        min_heap.push(nums[i]);
    }
}

double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
    /*
     Purpose: Find the median between the two sorted vectors
     Input:
        - nums1: A sorted vector of integers
        - nums2: A sorted vector of integers
     Output: A double representing the median between both input vectors
    */
    priority_queue<int,vector<int>,greater<int>> min_heap;
    int size = (int)nums1.size() + (int)nums2.size();
    
    createPriorityQueue4(nums1, min_heap);
    createPriorityQueue4(nums2, min_heap);
    
    while(min_heap.size() != size/2 + 1) min_heap.pop();
    
    double val = min_heap.top();
    
    if(size%2 == 0){
        min_heap.pop();
        val += min_heap.top();
        val /= 2;
    }
    
    return val;
}

/*
 6. ZigZag Conversion - Medium
 The string "PAYPALISHIRING" is written in a zigzag patter on a given number
 of rows like this: (you may want to display this pattern in a fixed font for
 legibility)

 P       A       H       N
 A   P   L   S   I   I   G
 Y       I       R

 And then read the line by line: "PAHNAPLSIIGYIR"

 Big(O) -> O(n), where n = size of the string
 Memory -> O(n), where n = size of the string
*/

void switchDirection(const int numRows, int &rowCount, bool &up){
    /*
     Purpose: Figure out if we need to switch the direction
     Input:
        - numRows: An integer representing the number of rows
        - rowCount: An integer of what row to look at
        - up: A boolean representing if we should be going up or down the rows
     Output: No output but this function changes the values of the variables via
             pass-by-reference
            - rowCount: An integer that becomes 1 if rowCount is negative and to
                        numRows - 2 if rowCount is equal to or greater than numRows
            - up: A boolean that changes to true if rowCount is negative and to false
                  if rowCount is equal to or greater than numRows
    */
    if(rowCount < 0){
        up = true;
        rowCount = 1;
    }else if(rowCount >= numRows){
        up = false;
        rowCount = numRows - 2;
    }
}

void createVector6(vector<string> &letters, const int numRows, const string s){
    /*
     Purpose: Populate the input vector to
     Input:
        - letters: A string vector of size numRows
        - numRows: An integer representing the number of rows
        - s: A string of what we are supposed to separate
     Output: No output but this function changes the values of the variable via
             pass-by-reference
            - letters: A string vector. The function parses through s and appends
                       the character into the appropriate rowCount of the vector
    */
    int rowCount = 0;
    bool up = true;
    
    for(int index = 0; index < s.length(); index++){
        letters[rowCount] += s[index];
        
        if(up) rowCount++;
        else rowCount--;
        
        switchDirection(numRows, rowCount, up);
    }
    
}

string convert(string s, int numRows) {
    /*
     Purpose: Convert the string s to its zigZag form
     Input:
        - s: A string of what we are supposed to separate
        - numRows: An integer of how many rows to separate the input string
     Output: A string of what the input string looks like once separated by numRows
    */
    if(numRows < 2) return s;
    
    vector<string> letters(numRows);
    createVector6(letters, numRows, s);
    
    string zigZag = "";
    for(int i = 0; i < numRows; i++){
        zigZag += letters[i];
    }
    
    return zigZag;
}

/*
 7. Reverse Integer - Easy

 Given a 32-bit signed integer, reverse digits of an integer.

 Note:
 Assume we are dealing with an environment which could only store integers
 within the 32-bit signed integer range: [-2^31, 2^31 - 1]. For the purpose of
 this problem, assume that your function returns 0 when the reverse integer
 overflows.

 Big(O) -> O(log(x))
 Memory -> O(1)
*/

int reverse(int x) {
    /*
     Purpose: To reverse the integer of x without converting it to a string
     Input: An integer
     Output: The reversed integer of x
    */
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

/*
 11. Container With Most Water - Medium

 Given n non-negative integers  a1, a2, ..., an, where each represents a point at
 coordinate (i, ai). n vertical lines are drawn such that the two endpoints of
 line i is at (i, ai) and (i, 0). Find the two lines, which together with x-axis
 forms a container, such that the container contains the most water.

 Note: You may not slant the container and n is at least 2.

 Big(O) -> O(n), where n = size of the array
 Memory -> O(1)

 ********* See StefanPochmann's post for a thorough explanation *********
*/

int maxArea(vector<int>& height) {
    /*
     Purpose: Find the maximum amount of water that can be filled in a container
              using the integer values in height
     Input: A vector of integers where each index reprsents a height
     Output: An integer of the maximum amount of water that can be contained
             using the different heights of the input vector
    */
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

/*
 41. First Missing Positive - Hard

 Given an unsorted integer array, find the smallest missing positive integer.

 Note:
 Your algorithm should run in O(n) time and uses constant extra space.

 Big(O) -> O(n), where n = size of the array
 Memory -> O(1)
*/

void moveValues(vector<int> &arr, const int size){
    /*
     Purpose: Move values around arr so that arr[i] == i + 1
     Input:
        - arr: An unsorted vector of integers
        - size: An integer denoting the size of arr
     Output: No output but this function changes arr via pass-by-reference. This
             function swaps values in arr from least to greatest. Ideally, the
             value at i should be i + 1.
    */
    for(int i = 0; i < size; i++){
        while(arr[i] > 0 && arr[i] < size && arr[arr[i] - 1] != arr[i]){
            swap(arr[i], arr[arr[i] - 1]);
        }
    }
}

int findNumber(const vector<int> arr, const int size){
    /*
     Purpose: Find the first missing positive integer in arr
     Input:
        - arr: A sorted vector of integers from least to greatest
        - size: An integer denoting the size of arr
     Output: An integer of the first missing positive number
    */
    for(int i = 0; i < size; i++){
        if(arr[i] != i + 1) return i + 1;
    }
    
    return size + 1;
}

int firstMissingPositive(vector<int>& nums) {
    /*
     Purpose: Find the first missing positive integer in nums
     Input: An unsorted vector of integers
     Output: An integer of the first missing positive number
    */
    int size = (int)nums.size();
    moveValues(nums, size);
    
    return findNumber(nums, size);
}

/*
 48. Rotate Image - Medium

 You are given an n x n 2D matrix representing an image.

 Rotate the image by 90 degrees (clockwise).

 Note:
 You have to rotate the image in-place, which means you have to modify the
 input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

 Big(O) -> O(n^2), where n = one dimension of the 2D vector
 Memory -> O(1)
*/

void switchValues(vector<vector<int>>& matrix, const int a, const int b){
    /*
     Purpose: Switches values of the current four sides of the box
     Input:
        - matrix: A 2D vector of integers representing an image
        - a: An integer
        - b: An integer
     Output: No output but changes the current perimeter within matrix
    */
    for(int i = 0; i < (b - a); i++){
        swap(matrix[a][a + i], matrix[a + i][b]);
        swap(matrix[a][a + i], matrix[b][b - i]);
        swap(matrix[a][a + i], matrix[b - i][a]);
    }
}

void rotate(vector<vector<int>>& matrix) {
    /*
     Purpose: Rotates 2D vector of integers clockwise
     Input: A 2D vector of integers
     Output: No output but rotates the 2D vector clockwise one time
    */
    int a = 0;
    int b = (int)matrix.size() - 1;
    
    while(a < b){ // Start from the outer box and work in
        switchValues(matrix, a, b);
        a++;
        b--;
    }
}

/*
 53. Maximum Subarray - Easy
 
 Given an integer array nums, find the contiguous subarray (containing
 at least one number) which has the largest sum and return its sum.
 
 Example:
 Input: [-2, 1, -3, 4, -1, 2, 1, -5, 4]
 Output: 6
 Explanation: [4, -1, 2, 1] has the largest sum = 6.
 
 Follow up:
 If you have figured out the O(n) solution, try coding another solution
 using the divide and conquer approach, which is more subtle.
 
 Big(O) -> O(n), where n = the size of the vector
 Memory -> O(1)
*/

int maxSubArray(vector<int> &nums){
    int max = INT_MIN, sum = 0;
    for(int i = 0; i < nums.size(); i++){
        sum += nums[i];
        if(max < sum) max = sum;
        if(sum < 0) sum = 0;
    }
    return max;
}

/*
 70. Climbing Stairs - Easy

 You are climbing a stair case. It takes n steps to reach to the top.

 Each time you can either climb 1 or 2 steps. In how many distinctive ways can you
 climb to the top?

 Note: Given n will be a positive integer.

 Big(O) -> O(n)
 Memory -> O(1)
*/

int climbStairs(int n) {
    /*
     Purpose: Find the different ways we can reach the top
     Input: An integer of how many steps it takes to reach the top
     Output: An integer of the different ways it takes to reach the top if we
             only climb 1 or 2 steps at a time.
    */
    if(n < 3) return n;
    
    int first = 1, second = 2, differentWays = 0;
    
    for(int i = 2; i < n; i++){
        differentWays = first + second;
        first = second;
        second = differentWays;
    }
    
    return differentWays;
}

/*
 74. Search a 2D Matrix - Medium

 Write an efficient algorithm that searches for a value in an m x n matrix.
 This matrix has the following properties:
 - Integers in each row are sorted from left to right.
 - The first integer of each row is greater than the last integer of the
   previous row.

 Big(O) -> O(logm + logn), where m and n are the dimensions of the matrix
 Memory -> O(1)
*/

int findRow(vector<vector<int>> matrix, int target){
    /*
     Purpose: Find the row the target value is in using binary search
     Input:
        - matrix: A 2D vector of integers
        - target: An integer to find in the matrix
     Output: An integer of what row the target value is in
    */
    int topRow = 0, bottomRow = (int)matrix.size() - 1;
    
    while(topRow < bottomRow){
        int midRow = (topRow + bottomRow)/2;
        int leftVal = matrix[midRow][0];
        int rightVal = matrix[midRow][matrix[midRow].size() - 1];
        
        if(leftVal > target && rightVal > target) bottomRow = midRow - 1;
        else if(leftVal < target && rightVal < target) topRow = midRow + 1;
        else return midRow;
    }
    
    return topRow;
}

bool searchMatrix(vector<vector<int>>& matrix, int target) {
    /*
     Purpose: Find if the target value is in matrix using binary search
     Input:
        - matrix: A 2D vector of integers
        - target: An integer to find in the matrix
     Output: A boolean. True if target is in matrix, false otherwise
    */
    if(matrix.size() == 0) return false;
    
    int row = findRow(matrix, target);
    int left = 0, right = (int)matrix[row].size() - 1;
    
    while(left <= right){
        int mid = (left + right)/2;
        
        if(matrix[row][mid] < target) left = mid + 1;
        else if(matrix[row][mid] > target) right = mid - 1;
        else return true;
    }
    
    return false;
}

/*
 75. Sort Colors - Medium

 Given an array with n objects colored red, white or blue, sort them in-place
 so that objects of the same color are adjacents, with the colors in the order
 red, white and blue.

 Here, we will use the integers 0, 1, and 2 to represent the color red, white,
 and blue respectively.

 Note: You are not suppose to use the library's sort function for this problem.

 Big(O) -> O(n), where n = size of the array
 Memory -> O(1)
*/

void sortColors(vector<int>& nums) {
    /*
     Purpose: Sort nums
     Input: A vector of integers of values 0, 1, 2 which represent red, white,
            blue respectively
     Output: No output but this function changes nums via pass-by-reference. The
             vector should be sorted from 0, 1, to 2
    */
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

/*
 118. Pascal's Triangle - Easy

 Given a non-negative integer numRows, generate the first numRows of
 Pascal's Triangle.

 Big(O) -> O(n^2), where n = numRows
 Memory -> O(n^2), where n = numRows
*/

vector<vector<int>> generate(int numRows){
    /*
     Purpose: Generate an input number of rows of Pascal's Triangle
     Input: An integer of how many rows of Pascal's Triangle to calculate
     Output: A 2D vector of integers where each vector represents a row
             of Pascal's Triangle
    */
    if(numRows <= 0) return {};
    
    vector<vector<int>> pascalsTriangle = {{1}};
    
    for(int i = 1; i < numRows; i++){
        pascalsTriangle.push_back({1});
        for(int j = 1; j < i; j++){
            //  Calculate the sum of the previous two values
            int sum = pascalsTriangle[i - 1][j] + pascalsTriangle[i - 1][j - 1];
            pascalsTriangle[i].push_back(sum);
        }
        pascalsTriangle[i].push_back(1);
    }
    
    return pascalsTriangle;
}

/*
 119. Pascal's Triangle II - Easy
 
 Given a non-negative index k where k ≤ 33, return the kth index row of
 the Pascal's triangle.

 Note that the row index starts from 0.

 Big(O) -> O(n), where n = rowIndex
 Memory -> O(n), where n = rowIndex
*/

vector<int> getRow(int rowIndex) {
    /*
     Purpose: Generate the input row of Pascal's Triangle
     Input: An integer of which row to calculate of Pascal's Triangle
     Output: A vector of integers of the rowIndex of Pascal's Triangle
    */
    vector<int> pascalsTriangle(rowIndex + 1, 1); // set entire row to 1
    long value = 1; // Needs to be long because after a certain value, int won't do
    
    for (int i = 1; i < rowIndex; i++) {
        value = value * (rowIndex - i + 1) / i;
        pascalsTriangle[i] = (int)value;
    }
    
    return pascalsTriangle;
}

/*
 153. Find Minimum in Rotated Sorted Array - Medium

 Suppose an array sorted in ascending order is rotated at some pivot point unknown
 to you beforehand.
 
 (i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).

 Find the minimum element.

 You may assume no duplicate exists in the array.

 Big(O) -> O(logn), where n = size of the array
 Memory -> O(1)
*/

int findMin(vector<int>& nums) {
    /*
     Purpose: Find the minimum element in a rotated sorted vector
     Input: A rotated sorted vector of integers
     Output: An integer representing the minimum element in a sorted vector
    */
    int start = 0, end = (int)nums.size() - 1;
    
    while (start < end) {
        //  If the left is less than the right, we know this portion is sorted
        //      and can return nums[start]
        if (nums[start] < nums[end]) return nums[start];
        
        //  Begin looking at the middle between start and end
        int mid = (start + end)/2;
        
        if (nums[mid] >= nums[start]) start = mid + 1;
        else end = mid;
    }
    
    return nums[start];
}

/*
 154. Find Minimum in Rotated Sorted Array II - Hard

 Suppose an array sorted in ascending order is rotated at some pivot
 unknown to you beforehand.

 (i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).

 Find the minimum element.

 The array may contain duplicates.

 Note:
 - This is a follow up problem to 153. Find the Minimum in Rotated Sorted Array.
 - Would allow duplicates affect the run-time complexity? How and why?

 Big(O) -> O(n), where n = the size of the vector
 Memory -> O(1)

 ********* Cannot do a binary search like we did in 153 because of the addition
 of duplicate values. There could be a case like [10,1,10,10,10] which would
 be difficult to determine which half to look at because nums[start] == nums[end]
 == nums[middle]. *********
*/

int findMinII(vector<int>& nums) {
    /*
     Purpose: Find the minimum element in the rotated sorted vector
              that may have duplicate values
     Input: A rotated sorted vector of integers that could have duplicates
     Output: An integer representing the minimum element in the input vector
    */
    for(int i = 1; i < nums.size(); i++){
        //  Assume first element is minimum element
        //  Once a smaller element is found, return value
        if(nums[0] > nums[i]) return nums[i];
    }
    
    //  if we exit for loop, this means the first element is smallest
    return nums[0];
}

/*
 169. Majority Element - Easy
 
 Given an array of size n, find the majority element. The majority element is the
 element that appears more than n/2 times.

 You may assume that the array is non-empty and the majority element alaways
 exist in the array.

 Big(O) -> O(n), where n = size of the array
 Memory -> O(n), where n = size of the array
*/

int majorityElement(vector<int>& nums){
    /*
     Purpose: Find the integer that occurs more than half the size of nums
     Input: A vector of integers
     Output: An integer from nums where its occurence appears more than half the
             size of the vector
    */
    int n = (int)nums.size();
    unordered_map<int,int> numToFrequency;
    
    for(int i = 0; i < n; i++){
        numToFrequency[nums[i]]++;
        if(numToFrequency[nums[i]] > n/2) return nums[i];
    }
    
    return -1; // This should never be reached since the majority element will exist
}

/*
 200. Number of Islands - Medium

 Given a 2d grid of '1's (land) and '0's (water), count the number of islands. An
 island is surrounded by water and is formed by connecting adjacent lands
 horizontally or vertically. You may assume all four edges of the grid are all
 surrounded by water.

 Big(O) -> O(n x m), where n and m are the dimensions of the 2d array
 Memory -> O(1)

 ********* This solution originially passed all cases. When I retested it, it kept
 failing on one test case. To fix this issue, replace the 'valid' function call in
 'findSurroundingLand' with the criteria check used in 'valid'. I'm not sure
 why this caused the one test case to fail as the call to 'valid' should be
 constant. *********
*/

bool valid(const vector<vector<char>> grid, const int i, const int j){
    /*
     Purpose: Validate the i,j coordinates in grid
     Input:
        - grid: A 2D vector of characters representing a grid
        - i: An integer representing the x-coordinate of the grid
        - j: An integer representing the y-coordinate of the grid
     Output: A boolean that returns true if the i and j are within the grid
             and if that point in the grid equals to 1. Otherwise returns false
    */
    return (i >= 0 && i < grid.size() && j >= 0 && j < grid[i].size() && grid[i][j] == '1');
}

void findSurroundingLand(vector<vector<char>> &grid, const int i, const int j){
    /*
     Purpose: Find the surrounding land
     Input:
        - grid: A 2D vector of characters representing a grid
        - i: An integer representing the x-coordinate of the grid
        - j: An integer representing the y-coordinate of the grid
     Output: No output but if the i and j point of the grid is valid, the function
             changes the value to 0 to ensure we do not visit this coordinate again.
    */
    if(!valid(grid, i, j)) return;
    
    grid[i][j] = '0'; // make sure not to visit this spot again
    
    findSurroundingLand(grid, i, j + 1); // down
    findSurroundingLand(grid, i, j - 1); // up
    findSurroundingLand(grid, i - 1, j); // left
    findSurroundingLand(grid, i + 1, j); // right
}

int numIslands(vector<vector<char>>& grid) {
    /*
     Purpose: Calculates how many islands are in the grid
     Input: A 2D vector of characters representing a grid
     Output: An integer of how many islands are found in the 2D vector
    */
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

/*
 203. Remove Linked List Elements - Easy
 
 Remove all elements from a linked list of integers that have value val.
 
 Example:
 Input: 1->2->6->3->4->5->6, val = 6
 Output: 1->2->3->4->5
 
 Big(O) -> O(n), where n = the size of the list
 Memory -> O(n), where n = the size of the list
*/

ListNode* removeElements(ListNode* head, int val) {
    ListNode* temp = new ListNode(0);
    ListNode* returnMe = temp;
    
    while(head != nullptr){
        if(head->val != val){
            temp->next = new ListNode(head->val);
            temp = temp->next;
        }
        head = head->next;
    }
    return returnMe->next;
}

/*
 209. Minimum Size Subarray Sum - Medium

 Given an array of n positive integers and a positive integer s, find the
 minimal length of a contiguous subarray of which the sum ≥ s. If there isn't
 one, return 0 instead.

 Big(O) -> O(n), where n = the size of the vector
 Memory -> O(1)

 ********* Here I used the chasing two pointer method. Have a running sum of
 using a faster index and once our running sum becomes greater than or equal
 to the target, subtract from the running sum using the slower index. The
 difference between the fast and slow pointers give the length of the subarray.
 *********
*/

int minSubArrayLen(int s, vector<int>& nums) {
    /*
     Purpose: Find the smallest subarray that sums up to a number greater
              than or equal to s
     Input:
        - s: An integer representing the minimum sum target we want to achieve
        - nums: A vector of integers
     Output: An integer representing the smallest subarray size whose sum is
             greater than or equal to s. If no such subarray exists, return 0.
    */
    int slow = 0, fast = 0;
    int n = (int)nums.size(), sum = 0;
    int smallestLength = INT_MAX;
    
    while(fast < n){
        sum += nums[fast];
        fast++;
        while(sum >= s){
            smallestLength = min(smallestLength, fast - slow);
            sum -= nums[slow];
            slow++;
        }
    }
    
    if(smallestLength == INT_MAX) return 0;
    else return smallestLength;
}

/*
 215. Kth Largest Element in an Array - Medium

 Find the kth largest element in an unsorted array. Note that it is the kth
 largest element in the sorted order, not the kth distinct element.

 Note:
 You may assume k is always valid, 1 ≤ k ≤ array's length.

 Big(O) -> O(klogk + (n-k)logk), where n = size of the array
 Memory -> O(k), where k = kth element. The heap should never be larger than this number.
*/

int findKthLargest(vector<int>& nums, int k) {
    /*
     Purpose: Find the kth largest element in nums once sorted
     Input:
        - nums: An unsorted vector of integers
        - k: An integer of which largest element to return once nums is sorted
     Output: An integer of the kth largest element in the sorted vector
    */
    priority_queue<int, vector<int>, greater<int>> min_heap;
    
    for(int i = 0; i < (int)nums.size(); i++){
        if(min_heap.size() < k) min_heap.push(nums[i]);
        else if(nums[i] > min_heap.top()){
            min_heap.pop();
            min_heap.push(nums[i]);
        }
    }
    
    return min_heap.top();
}

/*
 217. Contains Duplicate - Easy

 Given an array of integers, find if the array contains any duplicates.

 Your function should return true if any value appears at least twice in the
 array, and it should return false if every element is distinct.

 Big(O) -> O(n), where n = size of the array.
 Memory -> O(n), where n = size of the array.
*/

bool containsDuplicate(vector<int>& nums) {
    /*
     Purpose: Find the duplicate integer in the input vector
     Input: A vector of integers
     Output: A boolean where it returns true if a duplicate integer has been
             found in nums. False if otherwise.
    */
    unordered_map<int, int> numbersToOccurence;
    
    for(int i = 0; i < nums.size(); i++){
        numbersToOccurence[nums[i]]++;
        if(numbersToOccurence[nums[i]] > 1) return true;
    }
    
    return false;
}

/*
 230. Kth Smallest Element in a BST - Medium

 Given a binary search tree, write a function kthSmallest to find the kth
 smallest element in it.

 Note:
 You may assume k is always valid, 1 ≤ k ≤ BST's total elements.

 Big(O) -> O(klogk + (n-k)logk), where n = size of the tree
 Memory -> O(k), where k = kth element. The heap should never be larger than this number.
*/

void pushValuesToHeap(TreeNode* root, priority_queue<int> &max_heap, const int k){
    /*
     Purpose: Create a max_heap of integers from the binary tree
     Input:
        - root: A binary tree node of integers
        - max_heap: A heap of integers sorted from greatest to least
        - k: An integer of which the smallest element to return of the binary tree
    */
    if(root == nullptr) return;
    
    if(max_heap.size() < k) max_heap.push(root->val);
    else if(root->val < max_heap.top()){
        max_heap.pop();
        max_heap.push(root->val);
    }
    
    pushValuesToHeap(root->left, max_heap, k);
    pushValuesToHeap(root->right, max_heap, k);
}

int kthSmallest(TreeNode* root, int k){
    /*
     Purpose: Find the kth smallest element in the binary tree
     Input:
        - root: A binary tree node of integers
        - k: An integer of which the smallest element to return of the binary tree
     Output: An integer of the kth smallest element in the binary tree
    */
    priority_queue<int> max_heap;
    
    pushValuesToHeap(root, max_heap, k);
    
    return max_heap.top();
}

/*
 240. Search a 2D Matrix II - Medium

 Write an efficent algorithm that searches for a value in an m x n matrix.
 This matrix has the following properties:
 - Integers in each row are sorted in ascending from left to right.
 - Integers in each column are sorted in ascending from top to bottom.

 Big(O) -> O(m + n), where m and n are the dimensions of the 2D matrix
 Memory -> O(1)
*/

bool searchMatrixII(vector<vector<int>>& matrix, int target) {
    /*
     Purpose: Find if the target value is in the 2D vector
     Input:
        - matrix: A sorted 2D vector of integers. Each row and column are sorted
                  from least to greatest (left to right, top to bottom).
        - target: An integer to find in the 2D vector
     Output: A boolean that returns true/false if the target is in the 2D vector
    */
    if(matrix.size() == 0) return false;
    
    int x = 0, y = (int)matrix[0].size() - 1;
    
    while(x < matrix.size() && y >= 0){
        if(matrix[x][y] == target) return true;
        else if(matrix[x][y] < target) x++;
        else y--;
    }
    
    return false;
}

/*
 307. Range Sum Query - Mutable - Medium

 Given an integer array nums, find the sum of the elements between indices i
 and j (i ≤ j), inclusive.

 The update(i, val) function modifies nums by updating the element at index i to val.
 
 NumArray Big(O) -> O(n), where n = size of the array.
 update Big(O) -> O(n), where n = size of the array.
 sumRange Big(O) -> O(1)
 Memory -> O(n), where n = size of the array.
*/

class NumArray {
    vector<int> allSums, myArray;
public:
    NumArray(vector<int>& nums) { // Constructor
        /*
         Purpose: Create a NumArray object that has two private vectors
         Input: A vector of integers
         Output: No output but creates a NumArray object. This function populates
                 the private allSums and myArray integer vectors.
        */
        myArray = nums;
        int sum = 0;
        for(int i = 0; i < myArray.size(); i++){
            sum += myArray[i];
            allSums.push_back(sum);
        }
    }
    
    void update(int i, int val) {
        /*
         Purpose: Update myArray to have the new value at index i. Also update
                  allSums to reflect the new value
         Input:
            - i: An integer representing which index to update of myArray
            - val: An integer of what to update myArray[i] to
         Output: No output, just updates myArray[i] to val
        */
        for(int index = i; index < allSums.size(); index++){
            allSums[index] -= myArray[i];
            allSums[index] += val;
        }
        myArray[i] = val;
    }
    
    int sumRange(int i, int j) {
        /*
         Purpose: Returns the summation of myArray between i and j in O(1) time
         Input:
            - i: An integer representing the left endpoint
            - j: An integer representing the right endpoint
         Output: The summation of myArray between i and j
        */
        if(i == 0) return allSums[j];
        else return allSums[j] - allSums[i - 1];
    }
};

/*
 329. Longest Increasing Path in a Matrix - Hard

 Given an integer matrix, find the length of the longest increasing path.

 From each cell, you can either move to four directions: left, right, up or down.
 You may NOT move diagonally or move outside of the boundary (i.e. wrap-around
 is not allowed).

 Big(O) -> O(m x n), where m and n are the dimensions of the 2D vector
 Memory -> O(m x n), where m and n are the dimensions of the 2D vector

 ********* It is important to note that the longest path in the grid may not start
 at [0,0]. *********
*/

int findPath(vector<vector<int>> &matrix, int i, int j, int prevNum, vector<vector<int>> &pathTaken){
    /*
     Purpose: Find the longest path
     Input:
        - matrix: A 2D vector of integers
        - i: An integer representing the x coordinate in the grid
        - j: An integer representing the y coordinate in the grid
        - prevNum: An integer representing the previous number to compare to
        - pathTaken: A 2D vector of integers that holds the longest path from a
                     coordinate point in the original grid
     Output: An integer representing the longest path from matrix[i][j]
    */
    if(!(i >= 0 && i < matrix.size() && j >= 0 && j < matrix[i].size() && matrix[i][j] > prevNum)) return 0;
    
    //  If pathTaken[i][j] is not 0, it has an integer representing the longest path
    //  This makes it so we do not have to redo any work
    if(pathTaken[i][j] != 0) return pathTaken[i][j];
    
    int right = findPath(matrix, i + 1, j, matrix[i][j], pathTaken);
    int left = findPath(matrix, i - 1, j, matrix[i][j], pathTaken);
    int down = findPath(matrix, i, j + 1, matrix[i][j], pathTaken);
    int up = findPath(matrix, i, j - 1, matrix[i][j], pathTaken);
    
    //  Set furthest path to pathTaken[i][j]
    pathTaken[i][j] = max(right, max(left, max(down, up))) + 1;
    
    return pathTaken[i][j]; // At this point, pathTaken[i][j] has the longest path
}

int longestIncreasingPath(vector<vector<int>>& matrix) {
    /*
     Purpose: Find the longest path where the next number is greater than the previous
     Input: A 2D vector of integers representing a grid
     Output: An integer representing the longest path of increasing numbers.
    */
    if(matrix.size() == 0) return 0;
    
    int maxPath = 0;
    vector<vector<int>> pathTaken(matrix.size(), vector<int> (matrix[0].size(), 0));
    
    //  Need to loop through the whole grid
    for(int i = 0; i < matrix.size(); i++){
        for(int j = 0; j < matrix[i].size(); j++){
            maxPath = max(maxPath, findPath(matrix, i, j, -1, pathTaken));
        }
    }
    
    return maxPath;
}

/*
 344. Reverse String - Easy

 Write a function that reverses a string. The input string is given as an array
 of characters char[].
 
 Do not allocate extra space for another array, you must do this by modifying the
 input array in-place with O(1) extra memory.

 You may assume all the characters consist of printable ascii characters.

 Big(O) -> O(n), where n = size of the array
 Memory -> O(1)
*/

void reverseString(vector<char>& s) {
    /*
     Purpose: Reverses the vector
     Input: A vector of characters
     Output: No output. This function reverses the input vector
    */
    int beginning = 0, ending = (int)s.size() - 1;
    
    while(beginning < ending){
        swap(s[beginning], s[ending]);
        beginning++;
        ending--;
    }
}

/*
 378. Kth Smallest Element in a Sorted Matrix - Medium

 Given a n x n matrix where each of the rows and columns are sorted in ascending
 order, find the kth smallest element in the matrix.

 Note that it is the kth smallest element in the sorted order, not the kth distinct element.

 Big(O) -> O(klogk + (n-k)logk), where n = size of the matrix.
 Memory -> O(k), where k = kth element. The heap should never be larger than this number.
*/

int kthSmallest(vector<vector<int>>& matrix, int k) {
    /*
     Purpose: Find the kth smallest element in the 2D vector
     Input:
        - matrix: A 2D vector of integers
        - k: An integer of which the smallest element to return from matrix
     Output: The kth smallest element in matrix
    */
    priority_queue<int> max_heap;
    
    for(int i = (int)matrix.size() - 1; i >= 0; i--){
        for(int j = (int)matrix[0].size() - 1; j >= 0; j--){
            if(max_heap.size() < k) max_heap.push(matrix[i][j]);
            else if(matrix[i][j] < max_heap.top()){
                max_heap.pop();
                max_heap.push(matrix[i][j]);
            }
        }
    }
    
    return max_heap.top();
}

/*
 448. Find All Numbers Disappeared in an Array - Easy

 Given an array of integers where 1 ≤ a[i] ≤ n (n = size of array), some elements
 appear twice and others appear once.

 Find all the elements of [1, n] inclusive that do not appear in this array.

 Could you do it without extra space and in O(n) runtime? You may assume the
 returned list does not count as extra space.

 Big(O) -> O(n), where n = the size of the vector
 Memory -> O(1), assuming the returned list does not count as extra space
*/

void sortArray(vector<int> &nums){
    /*
     Purpose: Because we know all elements are between 1 and n inclusive, we can
              swap numbers into their corresponding index + 1 spot
     Input: An unsorted vector of integers
     Output: No output but this function swaps values in nums until
             nums[i] == i + 1
    */
    for(int i = 0; i < nums.size(); i++){
        while(nums[i] > 0 && nums[i] <= nums.size() && nums[nums[i] - 1] != nums[i]){
            swap(nums[i], nums[nums[i] - 1]);
        }
    }
}

void stepThroughArray(const vector<int> nums, vector<int> &missingNumbers){
    /*
     Purpose: Step through the vector and find where nums[i] != i + 1
     Input:
        - nums: For the most part, a sorted vector of integers
        - missingNumbers: An empty vector meant to hold missing numbers
     Output: No output but this function pushes numbers where the value does
             not equal the index + 1 in nums
    */
    for(int i = 0; i < nums.size(); i++){
        if(i + 1 != nums[i]){
            missingNumbers.push_back(i + 1);
        }
    }
}

vector<int> findDisappearedNumbers(vector<int>& nums) {
    /*
     Purpose: Finds the positive numbers that are missing in the input vector
     Input: An unsorted vector of integers
     Output: A vector of integers that contain the missing numbers in nums
    */
    vector<int> missingNumbers;
    sortArray(nums);
    stepThroughArray(nums, missingNumbers);
    return missingNumbers;
}

/*
 463. Island Perimeter - Easy

 You are given a map in form of a two-dimensional integer grid where 1 represents
 land and 0 represents water.

 Grid cells are connected horizontally/vertically (not diagonally). The grid
 is completely surrounded by water, and there is exactly one island (i.e., one or
 more connected land cells).

 The island doesn't have "lakes" (water inside that isn't connected to the water
 around the island). One cell is a square with side length 1. The grid is
 rectangular, width and height don't exceed 100. Determine the perimeter of the island.

 Big(O) -> O(m x n), where m and n represent the dimensions of the grid
 Memory -> O(1)
*/

int islandPerimeter(vector<vector<int>>& grid){
    /*
     Purpose: Find the perimeter of land that touches water
     Input: A 2D vector of integers of 0s and 1s that represent water and
            land respectively
     Output: An integer indicating how much land touches water
    */
    int land = 0, landOverlap = 0;
    
    for(int i = 0; i < grid.size(); i++){
        for(int j = 0; j < grid[i].size(); j++){
            if(grid[i][j] == 1){
                if(i != 0 && grid[i - 1][j] == 1) landOverlap++;
                if(j != 0 && grid[i][j - 1] == 1) landOverlap++;
                land++;
            }
        }
    }
    
    /*
     Multiply by 4 because that's the max perimeter a given land can have
     Multiply by 2 because if we find that a land touches another land,
        we need to take one side from both lands
    */
    return 4 * land - 2 * landOverlap;
}

/*
 509. Fibonacci Number - Easy

 The Fibonacci numbers, commonly denoted F(n) form a sequence, called the
 Fibonacci sequence, such that each number is the sum of the two preceding
 numbers, starting from 0 and 1. That is,
 F(0) = 0, F(1) = 1
 F(N) = F(N - 1) + F(N - 1), for N > 1

 Big(O) -> O(n), where n = N input.
 Memory -> O(1)
*/

int fib(int N) {
    /*
     Purpose: Calculate the Fibonacci summation of N
     Input: An integer to calculate the Fibonacci summation
     Output: An integer representing the Fibonacci summation of N
    */
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

/*
 530. Minimum Absolute Difference in BST - Easy

 Given a binary search tree with non-negative values, find the minimum absolute
 difference between values of any two nodes.

 Big(O) -> O(nlogn), where n = size of the tree
 Memory -> O(n), where n = size of the tree
*/

void traverseTree(const TreeNode* root, vector<int> &allNumbers){
    /*
     Purpose: Preorder traverse the tree and push the value to the vector
     Input:
        - root: A binary tree of integers
        - allNumbers: A vector of integers
     Output: No output but if the node has a value, then we push it to allNumbers
    */
    if(root == nullptr) return;
    
    allNumbers.push_back(root->val);
    
    traverseTree(root->left, allNumbers);
    traverseTree(root->right, allNumbers);
}

int getMinimumDifference(TreeNode* root) {
    /*
     Purpose: Calculate the absolute minimum difference between any two nodes
              of the binary tree
     Input: A binary tree of integers
     Output: An integer of the absolute minimum difference between any two nodes
             of the binary tree
    */
    vector<int> allNumbers;
    
    traverseTree(root, allNumbers);
    
    sort(allNumbers.begin(), allNumbers.end());
    
    int min = INT_MAX;
    
    for(int i = 1; i < (int)allNumbers.size(); i++){
        int absDiff = abs(allNumbers[i] - allNumbers[i - 1]);
        if(absDiff <= 1) return absDiff;
        else if(absDiff < min) min = absDiff;
    }
    
    return min;
}

/*
 605. Can Place Flowers - Easy

 Suppose you have a long flowerbed in which some of the plots are planted and
 some are not.
 However, flowers cannot be planted in adjacent plots - they would compete for
 water and both would die.

 Given a flowerbed (represented as an array containing 0 and 1, where 0 means
 empty and 1 means not empty), and a number n, return if n new flowers can be
 planted in it without violating the no-adjacent-flowers rule.

 Note:
 1. The input array won't violate no-adjacent-flowers rule.
 2. The input array size is in the range of [1, 20000].
 3. n is a non-negative integer which won't exceed the input array size.

 Big(O) -> O(n), where n = the size of the vector
 Memory -> O(1)
*/

bool canPlaceFlowers(vector<int>& flowerbed, int n) {
    /*
     Purpose: See if we can insert n amount of flowers into the vector
     Input:
        - flowerbed: A vector of integers of 0s and 1s where 0 represents dirt
                     and 1 represents a flower
        - n: An integer representing the number of flowers we wish to plant
     Output: A boolean if we can at least plant n amount of flowers
    */
    flowerbed.insert(flowerbed.begin(),0); // add 0 at the beginning
    flowerbed.push_back(0); // add 0 at the end
    int i = 1;
    while(i < flowerbed.size() - 1){
        // only decrease flower count n if sum equals to 0
        if(flowerbed[i - 1] + flowerbed[i] + flowerbed[i + 1] == 0){
            n--;
            i++; // increment here so we don't have to look at the same spot
        }
        i++;
    }
    //  It's not entirely clear in the original problem statement, but you can
    //  plant more than n times and still return true
    return n <= 0;
}

/*
 643. Maximum Average Subarray I - Easy

 Given an array consisting of n integers, find the contiguous subarray of given
 length k that has the maximum average value. And you need to output the maximum
 average value.

 Big(O) -> O(n), where n = size of the array.
 Memory -> O(1)
*/

double initialSum(const vector<int> &nums, const int k){
    /*
     Purpose: Calculate the initial sum from 0 to k-1
     Input:
        - nums: A vector of integers
        - k: An integer of the last index in nums to calculate the summation
     Output: A double of the summation from nums[0] to nums[k - 1]
    */
    double sum = 0;
    
    for(int i = 0; i < k; i++){
        sum += nums[i];
    }
    
    return sum;
}

double findMaxAverage(vector<int>& nums, int k) {
    /*
     Purpose: Find the maximum summation of a subarray of size k
     Input:
        - nums: A vector of integers
        - k: An integer representing the size of the subarray to calculate the
             the summation
     Output: A double of the maximum summation of the subarray divided by k
    */
    double sum = initialSum(nums, k);
    double max = sum; // Assume the initial sum is the maximum
    
    for(int i = 1; i < (int)nums.size() - k + 1; i++){
        sum -= nums[i - 1]; // Minus previous number
        sum += nums[i + k - 1]; // Add current number
        if(sum > max) max = sum;
    }
    
    return max/k;
}

/*
 695. Max Area of Island - Medium

 Given a non-empty 2D array grid of 0's and 1's, an island is a group of 1's
 (representing land) connected 4-directionally (horizontal or vertical.) You may
 assume all four edges of the grid are surrounded by water.

 Find the maximum area of an island in the given 2D array. (If there is no
 island, the maximum area is 0.)

 Big(O) -> O(n x m), where n and m are the dimensions of the grid.
 Memory -> O(1)

 ********* This solution originially passed all cases. When I retested it, it kept
 failing on one test case. To fix this issue, replace the 'valid' function call in
 'countArea' with the criteria check used in 'valid'. I'm not sure why this
 caused the one test case to fail as the call to 'valid' should be constant. *********
*/

bool valid(const vector<vector<int>> grid, const int x, const int y){
    /*
     Purpose: Validate the x,y coordinate within the grid
     Input:
        - grid: A 2D vector of integers representing a grid
        - x: An integer representing the x-coordinate of the grid
        - y: An integer representing the y-coordinate of the grid
     Output: A boolean that returns true if x and y are within the grid and if
             that coordinate equals 1. Returns false otherwise.
    */
    return x >= 0 && x < grid.size() && y >= 0 && y < grid[0].size() && grid[x][y] == 1;
}

int countArea(vector<vector<int>> &grid, const int x, const int y){
    /*
     Purpose: Count the surrounding area of a found island
     Input:
        - grid: A 2D vector of integers representing a grid
        - x: An integer representing the x-coordinate of the grid
        - y: An integer representing the y-coordinate of the grid
     Output: An integer of the size of the current area
    */
    if(!valid(grid, x, y)) return 0;
    
    grid[x][y] = 0; // ensure not to visit this point again
    
    int count = 1;
    count += countArea(grid, x + 1, y);
    count += countArea(grid, x - 1, y);
    count += countArea(grid, x, y + 1);
    count += countArea(grid, x, y - 1);
    
    return count;
}

int maxAreaOfIsland(vector<vector<int>>& grid) {
    /*
     Purpose: Find the largest island in the grid
     Input: A 2D vector of integers representing a grid
     Output: An integer of the maximum area in grid
    */
    int maxArea = 0;
    
    for(int i = 0; i < grid.size(); i++){
        for(int j = 0; j < grid[0].size(); j++){
            if(grid[i][j] == 1){
                int count = countArea(grid, i, j);
                if(count > maxArea) maxArea = count;
            }
        }
    }
    
    return maxArea;
}

/*
 704. Binary Search - Easy

 Given a sorted (in ascending order) integer array nums of n elements and a
 target value, write a function to search target in nums. If target exists, then
 return its index, otherwise return -1.

 Note:
 1. You may assume that all elements in nums are unique
 2. n will be in range of [1, 10000]
 3. The value of each element in nums will be in the range [-9999,9999]

 Big(O) -> O(logn), where n = size of the vector
 Memory -> O(1)
*/

int search(vector<int>& nums, int target) {
    /*
     Purpose: Find if a given number is in a vector
     Input:
        - nums: A vector of nums in ascending order
        - target: An integer that could be in nums
     Output: The index of target in nums. If its not in nums, return -1.
    */
    int start = 0, end = (int)nums.size() - 1;
    
    while(start <= end){
        int mid = (start + end)/2;
        
        if(nums[mid] < target) start = mid + 1;
        else if(nums[mid] > target) end = mid - 1;
        else return mid;
    }
    
    return -1;
}

/*
 746. Min Cost Climbing Stairs - Easy

 On a staircase, the i-th step has some non-negative cost cost[i] assigned (0 indexed).

 Once you pay the cost you can either climb one or two steps. You need to find
 minimum cost to reach the top of the floor, and you can either start from the
 step with index 0, or the step with index 1.

 Note:
 1. cost will have a length in range [2, 1000].
 2. Every cost[i] will be an integer in the range [0, 999].

 Big(O) -> O(n), where n = size of the array.
 Memory -> O(1)
*/

int minCostClimbingStairs(vector<int>& cost) {
    /*
     Purpose: Find the minimum cost it would take to climb up the stairs
              where we can only walk 1 or 2 steps at a time
     Input: A vector of integers
     Output: An integer of the least cost it takes to climb up the stairs
             where we are only able to climb 1 or 2 steps at a time
    */
    int oneBefore = 0, twoBefore = 0;
    
    for(int i = (int)cost.size() - 1; i >= 0; i--){
        int currentStair = cost[i] + min(oneBefore,twoBefore);
        twoBefore = oneBefore;
        oneBefore = currentStair;
    }
    
    return min(oneBefore, twoBefore);
}

/*
 771. Jewels and Stones - Easy

 You're given strings J representing the types of stones that are jewels, and S
 representing the stones you have. Each character in S is a type of stone you
 have. You want to know how many of the stones you have are also jewels.

 The letters in J are guaranteed distinct, and all characters in J and S are
 letters. Letters are case sensitive, so "a" is considered a different type of
 stone from "A".

 Note:
 - S and J will consist of letters nad have length at most 50.
 - The characters in J are distinct.

Big(O) -> O(j + s), where j and s are the sizes of strings 'J' and 'S'
          respectively.
 Memory -> O(1), we know 'stones' will never exceed more than 58 (2*26) entries.
           We also know the lengths of either string will not exceed 50, so we
           can consider the space complexity to be constant.
*/

int numJewelsInStones(string J, string S) {
    /*
     Purpose: Find how many stones are also jewels
     Input:
        - J: A string representing stones that are jewels that I own
        - S: A string representing stones that I own
     Output: An integer of how many stones I have that are also jewels
    */
    unordered_map<int,int> stones;
    
    for(int i = 0; i < S.length(); i++){
        stones[S[i]]++;
    }
    
    int stoneJewels = 0;
    for(int i = 0; i < J.length(); i++){
        stoneJewels += stones[J[i]];
    }
    
    return stoneJewels;
}

/*
 787. Cheapest Flights Within K Stops - Medium

 There are n cities connected by m flights. Each flights starts from city u and
 arrives at v with price w.

 Now given all the cities and flights, together with starting city src and the
 destination dst, your task is to find the cheapest price from src to dst with up
 to k stops. If there is no such route, output -1.

 Note:
 - The number of nodes n will be in range [1, 100], with nodes labeled from 0 to n - 1.
 - The size of flights will be in range [0, n * (n - 1) / 2].
 - The format of each flight will be (src, dst, price).
 - The price of each flight will be in the range [1, 10000].
 - k is in the range of [0, n - 1].
 - There will not be any duplicated flights or self cycles.

 Big(O) -> O(c*r), where c and r are the number of cities and number of routes
           between the cities
 Memory -> O(c), where c is the number of cities
*/

void flightPlan(unordered_map<int, vector<pair<int,int>>> &flightDestinations, const vector<vector<int>> flights){
    /*
     Purpose: Create a hash map for the different routes from a city and its cost
     Input:
        - flightDestinations: An empty hash map. Its key is an int representing
                              a city and its value is vector of pairs of ints.
                              First int of pair is the destination and the
                              second int is the cost
        - flights: A 2D vector of ints
     Output: No output but function populates flightDestinations via
             pass-by-reference
    */
    for(auto flight: flights){
        flightDestinations[flight[0]].emplace_back(flight[1], flight[2]);
    }
}

void flyRoute(unordered_map<int, vector<pair<int,int>>> flightDestinations, const int routes, queue<pair<int,int>> &differentRoutes, const int dst, int &cheapest){
    /*
     Purpose: Fly through all the route options
     Input:
        - flightDestinations: Hash map of cities to its destinations with
                              current cost
        - routes: An integer representing how many flights to look through
        - differentRoutes: A queue that holds all routes from the previous city
        - dst: An integer representing the destination city
        - cheapest: An integer representing the cheapest cost
     Output: No output but changes following variables via pass-by-reference
            - differentRoutes: Pops 'routes' values. Then pushes all other
                               possible destinations and their current cost
            - cheapest: An integer holding the cheapest option
    */
    for(int i = 0; i < routes; i++){
        auto currentRoute = differentRoutes.front();
        differentRoutes.pop();
        
        if(currentRoute.first == dst) cheapest = min(cheapest, currentRoute.second);
        
        for(auto x: flightDestinations[currentRoute.first]){
            if(currentRoute.second + x.second < cheapest){
                differentRoutes.push({x.first, currentRoute.second + x.second}); // Add price thus far
            }
        }
    }
    
}

int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int K) {
    /*
     Purpose: Find the cheapest route from source location to destination
     Input:
        - n: An integer representing the number of cities
        - flights: A 2D vector of flights. Each vector looks like {src, dst, cost}
        - src: An integer representing the starting city
        - dst: An integer representing the final city
        - K: An integer reprsenting how many "layovers" are allowed between
             src to dst city
     Output: An integer of the cheapest amount of money it takes to get from
             src to dst. If it is not possible to get to dst within K stops,
             then return -1;
    */
    unordered_map<int, vector<pair<int,int>>> flightDestinations;
    flightPlan(flightDestinations, flights);
    
    int cheapest = INT_MAX, stops = 0;
    queue<pair<int,int>> differentRoutes;
    differentRoutes.push({src, 0}); // Cost is 0 to start from initial city
    
    while(!differentRoutes.empty() && stops <= K + 1){
        int numRoutes = (int)differentRoutes.size();
        flyRoute(flightDestinations, numRoutes, differentRoutes, dst, cheapest);
        stops++;
    }
    
    if(cheapest == INT_MAX) return -1;
    else return cheapest;
}

/*
 807. Max Increase to Keep City Skyline - Medium

 In a 2 dimensional array grid, each value grid[i][j] represents the height of a
 building located there. We are allowed to increase the height of any number of
 buildings, by any amount (the amounts can be different for different buildings).
 Height 0 is considered to be a building as well.

 At the end, the "skyline" when viewed from all four directions of the grid, i.e.
 top, bottom, left, and right, must be the same as the skyline of the original
 grid. A city's skyline is the outer contour of the rectangles formed by all the
 buildings when viewed from a distance.

 See the following example.

 What is the maximum total sum that the height of the buildings can be increased?

 Example

 grid:
 3 0 8 4
 2 4 5 7
 9 2 6 3
 0 3 1 0

 The skyline viewed from top or bottom is: [9, 4, 8, 7]
 The skyline viewed from left or right is: [8, 7, 9, 3]

 gridNew:
 8 4 8 7
 7 4 7 7
 9 4 8 7
 3 3 3 3
 
 Output: 35

 Notes:
 - 1 < grid.length = grid[0].length <= 50
 - All heights grid[i][j] are in the range [0, 100]
 - All buildings in grid[i][j] occupy the entire grid cell, that is, they are 1 x
   1 x grid[i][j] rectangular prism.

 Big(O) -> O(n^2), where n is the product of the rows and columns of the grid.
 Memory -> O(n), where n is the size of a given row or column
*/

void findMaxes(const vector<vector<int>> grid, vector<int> &topBottom, vector<int> &leftRight, const int size){
    /*
     Purpose: Find the maximum heights of the skyline when looking at it from
              either top to bottom or left to right
     Input:
        - grid: A 2D vector of integers representing a grid
        - topBottom: An empty vector of integers
        - leftRight: An empty vector of integers
        - size: An integer representing the size of a row/column in grid
     Output: No output but changes the two input vectors via pass-by-reference
            - topBottom: Populates this vector of the maximum value when looking at
                         the grid from top to bottom
            - leftRight: Populates this vector of the maximum value when looking at
                         the grid from left to right
    */
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++) {
            topBottom[i] = max(topBottom[i], grid[i][j]);
            leftRight[j] = max(leftRight[j], grid[i][j]);
        }
    }
}

int heightIncrease(const vector<vector<int>> grid, const vector<int> topBottom, const vector<int> leftRight, const int size){
    /*
     Purpose: Calculate the total increase of each building if we set them to
              its maximum possible height so that the skyline remains the same
              when viewed from top to bottom and left to right
     Input:
        - grid: A 2D vector of integers representing a grid
        - topBottom: A vector of integers of the maximum value a point can be
                     when looking at the grid from top to bottom
        - leftRight: A vector of integers of the maximum value a point can be
                     when looking at the grid from left to right
        - size: An integer representing the size of a row/column in the grid
     Output: An integer of the total increase of each building in the grid
    */
    int increase = 0;
    
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            increase += min(topBottom[i], leftRight[j]) - grid[i][j];
        }
    }
    
    return increase;
}

int maxIncreaseKeepingSkyline(vector<vector<int>>& grid) {
    /*
     Purpose: Find the total height increase possible that doesn't change
              the skyline view
     Input: A 2D vector of integers representing a grid
     Output: An integer of the total increase of each building in the grid
    */
    int size = (int)grid.size();
    
    vector<int> topBottom(size), leftRight(size);
    findMaxes(grid, topBottom, leftRight, size);
    
    return heightIncrease(grid, topBottom, leftRight, size);
}

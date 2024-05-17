// Two Sum. Time Complexity = O(n), Space Complexity = O(n)
int numbers[] = {2,3,4,2,7};
int target = 10;

function int[] two_sum(int numbers[]; int target){

    dict map;
    for(int index=0; index<len(numbers); index++){
        int difference = target-numbers[index];
        int value = map[itoa(difference)];

        if(value){
            int result[] = array(value, index);
            return result;
            }

        map[itoa(numbers[index])] = index;
    }
}

int result[] = two_sum(numbers, target);
printf('>> The indexes are: %s\n', result);

// Three Sum. Time Complexity = O(n^2), Space Complexity = O(n)
int numbers[] = {1, 2, -2, -1, 3};

function int[] three_sum(int numbers[]) {

    for (int i=0; i<len(numbers)-2; i++) {
        dict map;
        for (int j=i+1; j<len(numbers); j++) {
            int sum = numbers[i] + numbers[j];
            int target = -sum;

            if (map[itoa(target)]) {
                int value = map[itoa(target)];
                int result[] = array(value, i, j);
                return result;
            }

            map[itoa(numbers[j])] = j;
        }
    }
}

int result[] = three_sum(numbers);
printf(">> The indices are: %s\n", result);


// Longest Common Prefix. Time Complexity = O(max_str_length), Space Complexity = O(1)
string names[] = array('floor', 'flower', 'flight');

function string longest_common_prefix(string names[]){
    string prefix;

    for(int i=1; i<len(names); i++){
        for(int n=0; n<len(names[i]) ;n++){
            if(names[i][n] != names[0][n]){
                break;
            }
            prefix = names[i][0:n+1];
        }
    }

    return prefix;
}

string result = longest_common_prefix(names);
printf('>> Longest common prefix is: %s\n', result);


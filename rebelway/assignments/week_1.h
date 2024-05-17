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


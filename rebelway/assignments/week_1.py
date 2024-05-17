# Question 3:
# Given an array of integers, return the indices of three numbers that add up to 0.
# example: [1, 2, -2, -1, 3] output = [0, 2, 3]

def threeSum(nums):
    # your code goes here
    pass


# Time and space complexity:

# Question 4:
# Given a singly linked list, reverse the nodes of the linked list
# Example 1: [1, 2, 3] output = [3, 2, 1]

class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next


def printList(head):
    while head:
        print(head.data)
        head = head.next


head = Node(1)
middle = Node(2)
tail = Node(3)

head.next = middle
middle.next = tail
tail.next = None

printList(head)


def reverseList(head):
    # your code goes here
    pass

# Time and space complexity:



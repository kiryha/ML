# Reverse linked list. Time and space complexity: O(n), O(1)
class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next


def print_list(head):

    while head:
        print(head.data)
        head = head.next


def reverse_list(head):

    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev


head = Node(1)
middle = Node(2)
tail = Node(3)

head.next = middle
middle.next = tail
tail.next = None

reversed_head = reverse_list(head)
print_list(reversed_head)

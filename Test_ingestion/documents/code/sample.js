// Sample JavaScript module for code-ingestion testing

function isPrime(num) {
  if (num < 2) return false;
  for (let i = 2; i <= Math.sqrt(num); i++) {
    if (num % i === 0) return false;
  }
  return true;
}

class TodoList {
  constructor() {
    this.items = [];
  }

  add(task) {
    this.items.push({ task, done: false });
  }

  complete(index) {
    if (this.items[index]) {
      this.items[index].done = true;
    }
  }
}

const list = new TodoList();
list.add("Write ingestion tests");
list.add("Deploy to Render");
console.log(list.items);

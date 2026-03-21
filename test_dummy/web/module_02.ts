interface User {
  id: number;
  name: string;
  email: string;
  role: "admin" | "user";
}

async function getUsers(): Promise<User[]> {
  const response = await fetch("/api/users");
  return response.json();
}

export function filterAdmins(users: User[]): User[] {
  return users.filter(u => u.role === "admin");
}
